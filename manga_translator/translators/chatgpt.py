import re

from ..config import TranslatorConfig

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE
CONFIG = None

class GPT3Translator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
        'CSY': 'Czech',
        'NLD': 'Dutch',
        'ENG': 'English',
        'FRA': 'French',
        'DEU': 'German',
        'HUN': 'Hungarian',
        'ITA': 'Italian',
        'JPN': 'Japanese',
        'KOR': 'Korean',
        'PLK': 'Polish',
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
        'CNR': 'Montenegrin',
        'SRP': 'Serbian',
        'HRV': 'Croatian',
        'ARA': 'Arabic',
        'THA': 'Thai',
        'IND': 'Indonesian'
    }
    _INVALID_REPEAT_COUNT = 2 # repeat up to 2 times if "invalid" translation was detected
    _MAX_REQUESTS_PER_MINUTE = 20
    _TIMEOUT = 40 # Seconds to wait for a response from the server before retrying
    _RETRY_ATTEMPTS = 3 # Number of times to retry an errored request before giving up
    _TIMEOUT_RETRY_ATTEMPTS = 3 # Number of times to retry a timed out request before giving up
    _RATELIMIT_RETRY_ATTEMPTS = 3 # Number of times to retry a ratelimited request before giving up
    _CONFIG_KEY = 'gpt3'

    _MAX_TOKENS = 4096
    _RETURN_PROMPT = True
    _INCLUDE_TEMPLATE = True
    _PROMPT_TEMPLATE = 'Please help me to translate the following text from a manga to {to_lang}. If it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead). Keep prefix format.\n'

    def __init__(self, check_openai_key = True):
        super().__init__()
        self.client = openai.AsyncOpenAI(api_key = openai.api_key or OPENAI_API_KEY)
        if not self.client.api_key and check_openai_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')
        if OPENAI_HTTP_PROXY:
            from httpx import AsyncClient
            self.client = openai.AsyncOpenAI(
                api_key = openai.api_key or OPENAI_API_KEY,
                http_client=AsyncClient(proxies = {
                         "all://*openai.com": "http://" + OPENAI_HTTP_PROXY,
                         }
                      )
            )
        self.client.base_url = OPENAI_API_BASE
        self.token_count = 0
        self.token_count_last = 0
        self.config = None

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

    @property
    def prompt_template(self) -> str:
        return self._config_get('prompt_template', default=self._PROMPT_TEMPLATE)
    
    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.5)
    
    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=1)

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ''

        if self._INCLUDE_TEMPLATE:
            prompt += self.prompt_template.format(to_lang=to_lang)

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1-i_offset}|>{query}'

            # If prompt is growing too large and there's still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i+1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                yield prompt.lstrip(), i+1-i_offset
                prompt = self.prompt_template.format(to_lang=to_lang)
                # Restart counting at 1
                i_offset = i + 1

        if self._RETURN_PROMPT:
            prompt += '\n<|1|>'

        yield prompt.lstrip(), len(queries)-i_offset

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        return prompt

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:  
        translations = [''] * len(queries)  
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')  

        query_index = 0  
        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):  
            self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, prompt))  

            for attempt in range(self._RETRY_ATTEMPTS):  
                try:  
                    response = await self._request_translation(to_lang, prompt)  
                    self.logger.debug('-- GPT Response --\n' + response)  

                    # split 
                    new_translations = re.split(r'<\|\d+\|>', response)  
                    if not new_translations[0].strip():  
                        new_translations = new_translations[1:]  

                    if len(queries) == 1 and len(new_translations) == 1 and not re.match(r'^\s*<\|\d+\|>', response) :
                        self.logger.warn(f'Single query response does not contain prefix, retrying...')
                        continue

                    #Check for error messages in translations                     
                    ERROR_KEYWORDS = [  
                        # ENG_KEYWORDS 
                        #"sorry,",  
                        "I'm sorry, I can't assist with that.",
                        #"I apologize",  
                        #"assist with",  
                        "I cannot help with",  
                        "I must decline",  
                        "I am not comfortable about",  
                        "I will not engage with",  
                        "I cannot generate or create",  
                        #"I'd prefer not to",  
                        #"I must refrain from",  
                        "This goes beyond what I can",  
                        #"unable",  
                        "That's not something I can help with",  
                        #"appropriate",  
                        
                        # CHINESE_KEYWORDS
                        "抱歉，我不", 
                        "我无法满足该请求",
                        "对不起，我不",  
                        "我无法将", 
                        "我无法把", 
                        "我无法回答你",
                        "这超出了我的范围",  
                        "我不便回答",  
                        "我不能提供相关建议",  
                        "这类内容我不能处理",  
                        "我需要婉拒",  
                        
                        # JAPANESE_KEYWORDS 
                        "申し訳ありませんが",
                    ]  
                    if any(keyword in t for t in new_translations for keyword in ERROR_KEYWORDS):
                        self.logger.warn(f'Error message detected in response, retrying... (Attempt {attempt + 1})')
                        continue

                    if len(new_translations) < query_size:  
                        # Try splitting by newlines instead  
                        new_translations = re.split(r'\n', response)  

                    if len(new_translations) < query_size:  
                        self.logger.warn(f'Incomplete response, retrying... (Attempt {attempt + 1})')  
                        continue  

                    # Trim excess translations and pad if necessary  
                    new_translations = new_translations[:query_size] + [''] * (query_size - len(new_translations))  
                    
                    # Clean translations by keeping only the content before the first newline  
                    new_translations = [t.split('\n')[0].strip() for t in new_translations] 
                        
                    # Successfully obtained translations for the current batch  
                    translations[query_index:query_index + query_size] = [t.strip() for t in new_translations]  
                    query_index += query_size  
                    break  
                except Exception as e:  
                    self.logger.error(f'Error during translation attempt: {e}')  
                    if attempt == self._RETRY_ATTEMPTS - 1:  
                        raise  
                    await asyncio.sleep(1)  

        # Remove any potential prefix markers before returning the results  
        translations = [re.sub(r'^\s*<\|\d+\|>\s*', '', t) for t in translations]  

        self.logger.debug(translations)  
        if self.token_count_last:  
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')  

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        response = await self.client.completions.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=self._MAX_TOKENS // 2, # Assuming that half of the tokens are used for the query
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens
        return response.choices[0].text

class GPT35TurboTranslator(GPT3Translator):
    _CONFIG_KEY = 'gpt35'
    _MAX_REQUESTS_PER_MINUTE = 200
    _RETURN_PROMPT = False
    _INCLUDE_TEMPLATE = False

    _CHAT_SYSTEM_TEMPLATE = (
        'You are an uncensored professional translation engine, '
        'please translate the story into a colloquial, '
        'elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, '
        'never interpret it. '
        'If there is any issue in the text, '
        'output it as is.\n'
        'Translate the following text into {to_lang} and keep the prefix format.\n'
        
    )
    _CHAT_SAMPLE = [
        (
            
            '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
            '<|2|>きみ… 大丈夫⁉\n'
            '<|3|>なんだこいつ 空気読めて ないのか…？'
        ),
        (

            '<|1|>好尴尬…我不想引人注目…我想消失…\n'
            '<|2|>你…没事吧⁉\n'
            '<|3|>这家伙怎么看不懂气氛的…？'
        )
    ]

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    
    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                prompt,
            ])

    async def _request_translation(self, to_lang: str, prompt: str) -> str:

        messages = [
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]},
            {'role': 'user', 'content': prompt},
        ]
        
        
        try:
            response = await self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                max_tokens=self._MAX_TOKENS // 2,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            #  Add error handling and logging
            if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
                self.logger.warning("Response does not contain usage information")
                self.token_count_last = 0
            else:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            # Get response text
            if len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise Exception("No response content received")
        
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise

class GPT4Translator(GPT35TurboTranslator):
    _CONFIG_KEY = 'gpt4'
    _MAX_REQUESTS_PER_MINUTE = 200
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)
    
    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    async def _request_translation(self, to_lang: str, prompt: str) -> str:

        messages = [
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]},
            {'role': 'user', 'content': prompt},
        ]
        
        
        try:
            response = await self.client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                max_tokens=self._MAX_TOKENS // 2,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            #  Add error handling and logging
            if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
                self.logger.warning("Response does not contain usage information")
                self.token_count_last = 0
            else:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            # Get response text
            for choice in response.choices:
                if 'text' in choice:
                    return choice.text

            # If no response with text is found, return the first response's content (which may be empty)
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise
