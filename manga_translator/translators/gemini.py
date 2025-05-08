import re
from google import genai
from google.genai import types

import asyncio
from typing import List
from .common import MissingAPIKeyException, InvalidServerResponse
from .keys import GEMINI_API_KEY, GEMINI_MODEL
from .common_gpt import CommonGPTTranslator, _CommonGPTTranslator_JSON


# Text Formatting:
# For Windows: enable ANSI escape code support
from colorama import init as initColorama

BOLD='\033[1m' # Bold text
NRML='\033[0m' # Revert to Normal formatting

class GeminiTranslator(CommonGPTTranslator):
    _INVALID_REPEAT_COUNT = 0  # 现在这个参数没意义了
    _MAX_REQUESTS_PER_MINUTE = 9999  # 无RPM限制
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    # Maximum token count for controlling the length of text processed
    _MAX_TOKENS = 8192

    # 将每个 prompt 限制为最大输出 tokens 的 50％。
    # （这是一个任意比率，用于解释语言之间的差异。）
    # 
    # Limit each prompt to 50% max output tokens. 
    # (This is an arbitrary ratio to account for variance between languages.)
    _MAX_TOKENS_IN = _MAX_TOKENS // 2

    # From: https://ai.google.dev/gemini-api/docs/models/gemini#available-languages
    '''
    _LANGUAGE_CODE_MAP= {
                            'ar': 'Arabic',
                            'bn': 'Bengali',
                            'bg': 'Bulgarian',
                            'zh': 'Chinese simplified and traditional',
                            'hr': 'Croatian',
                            'cs': 'Czech',
                            'da': 'Danish',
                            'nl': 'Dutch',
                            'en': 'English',
                            'et': 'Estonian',
                            'fi': 'Finnish',
                            'fr': 'French',
                            'de': 'German',
                            'el': 'Greek',
                            'iw': 'Hebrew',
                            'hi': 'Hindi',
                            'hu': 'Hungarian',
                            'id': 'Indonesian',
                            'it': 'Italian',
                            'ja': 'Japanese',
                            'ko': 'Korean',
                            'lv': 'Latvian',
                            'lt': 'Lithuanian',
                            'no': 'Norwegian',
                            'pl': 'Polish',
                            'pt': 'Portuguese',
                            'ro': 'Romanian',
                            'ru': 'Russian',
                            'sr': 'Serbian',
                            'sk': 'Slovak',
                            'sl': 'Slovenian',
                            'es': 'Spanish',
                            'sw': 'Swahili',
                            'sv': 'Swedish',
                            'th': 'Thai',
                            'tr': 'Turkish',
                            'uk': 'Ukrainian',
                            'vi': 'Vietnamese',
                        }
    '''

    _MIN_CACHE_TOKENS = 4096 # Minimum tokens required to use Context Cache
                            # Source: https://ai.google.dev/gemini-api/docs/caching?lang=python#considerations
    
    _CACHE_TTL = 3600 # Set the Context Cache lifespan (seconds)
    _CACHE_TTL_BUFFER = 300 # Refresh the Context Cache once current time is within this many seconds of expiring

    def __init__(self):
        # ConfigGPT 的初始化
        # ConfigGPT initialization 
        _CONFIG_KEY = 'gemini.' + GEMINI_MODEL
        CommonGPTTranslator.__init__(self, config_key=_CONFIG_KEY)

        # Initialize colorama for ANSI encoding support
        #   (Only required on Windows)
        initColorama()

        # By default: Do not assume Context Cache support
        self._canUseCache = False
        self.cached_content = None
        self.templateCache = None

        # Dict for storing values to print to logger
        self.cachedVals={None}

        if not GEMINI_API_KEY:
            raise MissingAPIKeyException(
                        'Please set the GEMINI_API_KEY environment variable '
                        'before using the Gemini translator.'
                    )

        self.client = genai.Client(api_key=GEMINI_API_KEY)

        try:
            model_list=self.client.models.list()
            #convert pager object to list
            model_list = list(model_list)
        except genai.errors.APIError as genai_err:
            raise InvalidServerResponse(
                        'GEMINI_API_KEY was found, but the API failed to connect.\n.' +
                        f'The following error was caught:\n{genai_err}'
                    )
        except Exception as e:
            self.logger.error(
                        'GEMINI_API_KEY was found, but an unknown error was encountered during initial setup.\n.' +
                        f'The following error was caught:\n{genai_err}'
                    )
            raise Exception(f"Model: '{GEMINI_MODEL}' was not found in the model list.\n" +
                                "Please ensure you set the key: GEMINI_MODEL to one of the following values:"
                                '\n'.join(mName for mName in model_names))

        '''
        Start Section:
            Validate `GEMINI_MODEL` specification and determine supported capabilities.
        '''
        model_names = [aModel.name.lstrip('models/') for aModel in model_list]
        if  f"{GEMINI_MODEL}" not in model_names:
            self.logger.error(f"Model: '{GEMINI_MODEL}' was not found in the model list.\n" +
                                "Please ensure you set the key: GEMINI_MODEL to one of the following values:"
                            )
            self.logger.error('\n'.join(mName for mName in model_names))

            raise
        
        # Use index of model name to get full model info
        model_info = model_list[model_names.index(GEMINI_MODEL)]
        

        
        def canCache(model_list, model_info) -> bool:
            """
            Checks if the selected model is capable of using context caching.
            Made into a function purely to help with code readability.
            """
            # List of models that support content caching:
            canCacheModels=[m.name.lstrip('models/')
                            for m in model_list
                                if 'createCachedContent' in m.supported_actions
                        ]
            
            # If the model supports Context Caching: Enable
            # Else: Inform the user, list supported models
            if 'createCachedContent' in model_info.supported_actions:
                return True
            else:
                MSG= "ALERT:\n" + \
                    f"Model '{GEMINI_MODEL}' does not support Context Caching.\n" + \
                    "Context Caching allows you reduce token usage by storing " + \
                    "and reusing `System Prompt` and `Chat Samples`, " + \
                    "rather than re-sending it each time.\n\n" + \
                    "If you wish to use this feature, " + \
                    "set the GEMINI_MODEL key to one of the following values:\n\n" + \
                    '\n'.join(canCacheModels) + '\n\n' + \
                    "Note that the model name must be set to the precise version-name listed.\n" + \
                    "\te.g. 'gemini-1.5-flash-001' rather than 'gemini-1.5-flash'\n"
                self.logger.warning(MSG)

        self._canUseCache = canCache(model_list, model_info)

        
        self._MAX_TOKENS = model_info.output_token_limit
        self._MAX_TOKENS_IN = self._MAX_TOKENS // 2


        ''''
            Set all `safety_settings` to 'Block None'

            Taken from official Google example code:
                Books contain all sorts of fictional or historical descriptions, 
                    some of them rather literal and might cause the model to stop 
                    from performing translation query. 
                To prevent some of those exceptions users are able to change 
                    `safety_setting` from default to more open approach.
            
            -- https://github.com/google-gemini/cookbook/blob/main/examples/Translate_a_Public_Domain_Book.ipynb
        '''
        self.safety_settings = [    {
                                        "category": "HARM_CATEGORY_HARASSMENT",
                                        "threshold": "BLOCK_NONE",
                                    }, {
                                        "category": "HARM_CATEGORY_HATE_SPEECH",
                                        "threshold": "BLOCK_NONE",
                                    }, {
                                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                        "threshold": "BLOCK_NONE",
                                    }, {
                                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                        "threshold": "BLOCK_NONE",
                                    }
                                ]
        self.token_count = 0
        self.token_count_last = 0 
        self.config = None

    @property
    def useCache(self) -> bool:
        if self._canUseCache:
            try:
                if self._needRecache:
                    self._createContext(to_lang=self.to_lang)
                
                return True
            
            except Exception as e:
                self.logger.warning(
                    f"\nContext Cache is supported on this model, but the cache could not be created.\n"
                    f"The following error was encountered when attempting to create Context Cache:\n{e}\n\n"
                    f"The most likely cause is that context contents (`System Prompt` + `Chat Samples`) does not the meet the minimum token length for the model.\n"
                    "Context Caching will be disabled. If you wish to use caching: Try using Gemini 1.5 or increase `System Prompt` and/or `Chat Sample` size."
                )
                self._canUseCache = False

        return False

    def parse_args(self, args: CommonGPTTranslator):
        super().parse_args(args)
        
        # Initialize mode-specific components AFTER config is loaded
        if self.json_mode:
            self._init_json_mode()
        else:
            self._init_standard_mode()

    def _init_json_mode(self):
        """Activate JSON-specific behavior"""
        self._json_funcs = _GeminiTranslator_json(self)

        self._createContext = self._json_funcs._createContext
        self._request_translation = self._json_funcs._request_translation
        self._assemble_prompts = self._json_funcs._assemble_prompts
        self._parse_response = self._json_funcs._parse_response

    def _init_standard_mode(self):
        """Use default method implementations"""
        self._assemble_prompts = super()._assemble_prompts


    def count_tokens(self, text: str) -> int:
        # Uses the synchronous call (`client`) instead of asynchronous (`client.aio`)
        #   for compatibility with `common_gpt` 's `assemble_prompt`
        return self.client.models.count_tokens(model=GEMINI_MODEL, contents=text).total_tokens
    
    def _createContext(self, to_lang: str): 
        chatSamples=None
        sysTemplate=self.chat_system_template.format(to_lang=to_lang)
        
        # Store cached values for printing to logger:
        self.cachedVals={'System Prompt (Cached)': sysTemplate}

        # 如果需要先给出示例对话
        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            chatSamples=[
                types.Content(role='user',  parts=[types.Part.from_text(text=lang_chat_samples[0])]),
                types.Content(role='model', parts=[types.Part.from_text(text=lang_chat_samples[1])]),
            ]
            self.cachedVals['Sample (Cached): User'] = lang_chat_samples[0]
            self.cachedVals['Sample (Cached): Model'] = lang_chat_samples[1]

        self.templateCache = self.client.caches.create(model=GEMINI_MODEL,
                                                        config=types.CreateCachedContentConfig(
                                                            contents=chatSamples,
                                                            system_instruction=sysTemplate,
                                                            display_name='TranslationCache',
                                                            ttl=f'{self._CACHE_TTL}s',
                                                        ),
                                                    )
        
    def _needRecache(self) -> bool:
        if not self.templateCache:
            return True

        # expire_time (as seconds) - now (as seconds)
        delta = (
                    # Get expire_time as unix timestamp
                    self.templateCache.expire_time.timestamp()
                    -
                    # Access `datetime.datetime` library through through the variable. 
                    # Get current time as unixtimestamp
                    self.templateCache.expire_time.now().timestamp()
            )
        
        # If cache expire_time is less than 5 minutes (300 seconds) in the future: return True
        return delta < self._CACHE_TTL_BUFFER

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:  
        self.to_lang=to_lang # Export `to_lang`
        translations = [''] * len(queries)  
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')  
        MAX_SPLIT_ATTEMPTS = 5  # Default max split attempts  
        RETRY_ATTEMPTS = self._RETRY_ATTEMPTS  

        async def translate_batch(prompt_queries, prompt_query_indices, split_level=0):  
            nonlocal MAX_SPLIT_ATTEMPTS
            split_prefix = ' (split)' if split_level > 0 else ''  

            # Assemble prompt for the current batch  
            prompt, query_size = self._assemble_prompts(from_lang, to_lang, prompt_queries).__next__()

            for attempt in range(RETRY_ATTEMPTS):  
                try:  
                    # Get the response (synchronously)
                    response = await self._request_translation(to_lang, prompt)  
                    try:
                        new_translations = self._parse_response(response, prompt_queries)
                    except Warning as w:
                        self.logger.warning(w)
                        self.logger.warning(f"Retrying...(Attempt {attempt + 1})")
                        continue
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error(f"Retrying...(Attempt {attempt + 1})")
                        continue

                    if len(new_translations) < query_size:  
                        # Try splitting by newlines instead  
                        new_translations = re.split(r'\n', response)  

                    if len(new_translations) < query_size:  
                        remaining_attempts = RETRY_ATTEMPTS - attempt - 1  
                        self.logger.warning(f'Incomplete response, remaining {remaining_attempts} time(s) before splitting the translation.')  
                        continue  
                    
                    # Trim excess translations and pad if necessary  
                    new_translations = new_translations[:query_size] + [''] * (query_size - len(new_translations))  
                    # Clean translations by keeping only the content before the first newline  
                    new_translations = [t.split('\n')[0].strip() for t in new_translations]  
                    # Remove any potential prefix markers  
                    new_translations = [re.sub(r'^\s*<\|\d+\|>\s*', '', t) for t in new_translations]  
                    # Check if any translations are empty  
                    if any(not t.strip() for t in new_translations):  
                        self.logger.warning(f'Empty translations detected. Resplitting the batch.') 
                        break  # Exit retry loop and trigger split logic below 

                    # Store the translations in the correct indices  
                    for idx, translation in zip(prompt_query_indices, new_translations):  
                        translations[idx] = translation  

                    # Log progress  
                    self.logger.info(f'Batch translated: {len([t for t in translations if t])}/{len(queries)} completed.')  
                    self.logger.debug(f'Completed translations: {[t if t else queries[i] for i, t in enumerate(translations)]}')        
                    return True  # Successfully translated this batch  
                    
                except genai.errors.APIError:  
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error(
                            'Gemini encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warning(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)
                except Exception as e:  
                    self.logger.error(f'Error during translation attempt: {e}')  
                    if attempt == RETRY_ATTEMPTS - 1:  
                        raise  
                    await asyncio.sleep(1)  

            # If retries exhausted and still not successful, proceed to split if allowed  
            if split_level < MAX_SPLIT_ATTEMPTS:  
                if split_level == 0:  
                    self.logger.warning('Retry limit reached. Starting to split the translation batch.')  
                else:  
                    self.logger.warning('Further splitting the translation batch due to persistent errors.')  
                mid_index = len(prompt_queries) // 2  
                futures = []  
                # Split the batch into two halves  
                for sub_queries, sub_indices in [   
                    (prompt_queries[:mid_index], prompt_query_indices[:mid_index]),  
                    (prompt_queries[mid_index:], prompt_query_indices[mid_index:]),  
                ]:  
                    if sub_queries:  
                        futures.append(translate_batch(sub_queries, sub_indices, split_level + 1))  
                results = await asyncio.gather(*futures)  
                return all(results)  
            else:  
                self.logger.error('Maximum split attempts reached. Unable to translate the following queries:')  
                for idx in prompt_query_indices:  
                    self.logger.error(f'Query: {queries[idx]}')  
                return False  # Indicate failure for this batch   

        # Begin translation process  
        prompt_queries = queries  
        prompt_query_indices = list(range(len(queries)))  
        await translate_batch(prompt_queries, prompt_query_indices)  

        self.logger.debug(translations)  
        if self.token_count_last:  
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')  
        return translations

    def formatLog(self, vals: dict) -> str:
        return '\n---\n'.join(f"\n{BOLD}{aKey}{NRML}:\n{aVal}" 
                                for aKey, aVal in vals.items()
                            )

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        config_kwargs = {
                            'safety_settings': self.safety_settings,
                            'top_p': self.top_p,
                            'temperature': self.temperature,
                        }
        
        messages=[]

        # Store values to be printed to logger
        loggerVals={}
        if self.useCache:
            config_kwargs['cached_content'] = self.templateCache.name
            
            loggerVals = self.cachedVals.copy()
        else:
            config_kwargs['system_instruction'] = self.chat_system_template.format(to_lang=to_lang)
            loggerVals = {'System Prompt': config_kwargs['system_instruction']}

            # 如果需要先给出示例对话
            # Add chat samples if available
            lang_chat_samples = self.get_chat_sample(to_lang)
            if lang_chat_samples:
                messages=[
                    types.Content(role='user',  parts=[types.Part.from_text(text=lang_chat_samples[0])]),
                    types.Content(role='model', parts=[types.Part.from_text(text=lang_chat_samples[1])])
                ]

                loggerVals['Sample: User'] = lang_chat_samples[0],
                loggerVals['Sample: Model'] = lang_chat_samples[1]


        messages.append(types.Content(role='user',  parts=[types.Part.from_text(text=prompt)]))
        loggerVals['Input'] = prompt

        self.logger.debug(  '-- GPT Prompt --\n' +
                            self.formatLog(loggerVals) +
                            '\n------------'
                        )

        response = await self.client.aio.models.generate_content(
                                                model=GEMINI_MODEL,
                                                contents=messages,
                                                config=types.GenerateContentConfig(
                                                            **config_kwargs
                                                        )
                                            )

        try:
            if not hasattr(response, 'usage_metadata'):
                self.logger.warning("Response does not contain usage information")
                self.token_count_last = 0
            else:
                self.token_count += response.usage_metadata.prompt_token_count
                self.token_count_last = response.usage_metadata.total_token_count
            
            self.logger.debug(f'-- GPT Response --\n' + response.text)

            return response.text
        except Exception as ex:
            self.logger.error(f"Error in _request_translation: {str(ex)}")
            raise ex



class _GeminiTranslator_json (_CommonGPTTranslator_JSON):
    from .config_gpt import TranslationList
    import json

    """Internal helper class for JSON mode logic"""
    def __init__(self, translator: GeminiTranslator):
        super().__init__(translator)
        self.translator = translator

        # For conveniance: Simplify logger calls:
        self.logger = self.translator.logger 

    async def _createContext(self, to_lang: str):
        JSON_Samples=[]
        sysTemplate=self.translator.chat_system_template.format(to_lang=to_lang)

        # Store cached values for printing to logger:
        self.cachedVals={'System Prompt (Cached)': sysTemplate}

        # 如果需要先给出示例对话
        # Add chat samples if available
        lang_JSON_samples = self.translator.get_json_sample(to_lang)
        if lang_JSON_samples:
            JSON_Samples=[
                types.Content(role='user',  parts=[types.Part.from_text(text=lang_JSON_samples[0].model_dump_json())]),
                types.Content(role='model', parts=[types.Part.from_text(text=lang_JSON_samples[1].model_dump_json())]),
            ]

            self.cachedVals['Sample (Cached): User'] = self.ppJSON(lang_JSON_samples[0].model_dump_json())
            self.cachedVals['Sample (Cached): Model'] = self.ppJSON(lang_JSON_samples[1].model_dump_json())

        self.templateCache = await self.translator.client.aio.caches.create(model=GEMINI_MODEL,
                                                                            config=types.CreateCachedContentConfig(
                                                                                contents=JSON_Samples,
                                                                                system_instruction=sysTemplate,
                                                                                display_name='TranslationCache_JSON',
                                                                                ttl=f'{self.translator._CACHE_TTL}s',
                                                                                ),
                                                                            )

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        config_kwargs = {
                            'safety_settings': self.translator.safety_settings,
                            'response_mime_type': 'application/json',
                            'response_schema': self.TranslationList,
                            'top_p': self.translator.top_p,
                            'temperature': self.translator.temperature,
                    }

        messages=[]

        # Store values to be printed to logger
        loggerVals={}
        if self.translator.useCache:
            config_kwargs['cached_content'] = self.templateCache.name
            loggerVals = self.cachedVals
        else:
            config_kwargs['system_instruction'] = self.translator.chat_system_template.format(to_lang=to_lang)
            loggerVals={'System Prompt': config_kwargs['system_instruction']}

            lang_JSON_samples = self.translator.get_json_sample(to_lang)
            if lang_JSON_samples:
                messages=[
                    types.Content(role='user',  parts=[types.Part.from_text(text=lang_JSON_samples[0].model_dump_json())]),
                    types.Content(role='model', parts=[types.Part.from_text(text=lang_JSON_samples[1].model_dump_json())]),
                ]

            loggerVals['Sample: User'] = lang_JSON_samples[0].model_dump_json(),
            loggerVals['Sample: Model'] = lang_JSON_samples[1].model_dump_json()


        messages.append(types.Content(role='user',  parts=[types.Part.from_text(text=prompt)]))
        
        loggerVals['Input'] = self.ppJSON(prompt)
        self.logger.debug(  '-- GPT Prompt --\n' +
                            self.translator.formatLog(loggerVals) +
                            '\n------------'
                        )
        
        response = await self.translator.client.aio.models.generate_content(model=GEMINI_MODEL,
                                                                            contents=messages,
                                                                            config=types.GenerateContentConfig(
                                                                                **config_kwargs
                                                                            )
                                                                        )

        try:
            if not hasattr(response, 'usage_metadata'):
                self.logger.warning("Response does not contain usage information")
                self.translator.token_count_last = 0
            else:
                self.translator.token_count += response.usage_metadata.prompt_token_count
                self.translator.token_count_last = response.usage_metadata.total_token_count

            self.logger.debug(  '-- GPT Response --\n' + 
                                self.ppJSON(response.text) + 
                                '\n------------\n'
                            )

            return response.text
        except Exception as ex:
            self.logger.error(f"Error in _request_translation: {str(ex)}")
            raise ex
