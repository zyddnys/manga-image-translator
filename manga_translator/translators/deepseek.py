import re

from ..config import TranslatorConfig

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List
from .common import CommonTranslator, MissingAPIKeyException
from .keys import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from .config_gpt import ConfigGPT
from .tokenizers.token_counters import deepseekTokenCounter


class DeepseekTranslator(ConfigGPT, CommonTranslator):
    _INVALID_REPEAT_COUNT = 0  # 现在这个参数没意义了
    _MAX_REQUESTS_PER_MINUTE = 9999  # 无RPM限制
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    # Maximum token count for controlling the length of text processed
    # 
    # 最大输出长度: 8K
    # MAX OUTPUT TOKENS: 8K
    # -- https://api-docs.deepseek.com/quick_start/pricing
    _MAX_TOKENS = 8000

    # 将每个 prompt 限制为最大输出 tokens 的 50％。
    # （这是一个任意比率，用于解释语言之间的差异。）
    # 
    # Limit each prompt to 50% max output tokens. 
    # (This is an arbitrary ratio to account for variance between languages.)
    _MAX_TOKENS_IN = _MAX_TOKENS // 2

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    def __init__(self, check_openai_key=True):
        # ConfigGPT 的初始化
        # ConfigGPT initialization 
        _CONFIG_KEY = 'deepseek.' + DEEPSEEK_MODEL
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        CommonTranslator.__init__(self)

        # Initialize the token counter
        tokenizer = deepseekTokenCounter()

        '''
        通过字符估计标记很困难，并且因语言而异:
        - 1 个英文字符 ≈ 0.3 个 token。
        - 1 个中文字符 ≈ 0.6 个 token。
        -- https://api-docs.deepseek.com/zh-cn/quick_start/token_usage
        
        因此：使用 deepseek 的 tokenizer 来准确计算 token 的数量。
        
        Estimating tokens by characters is tricky and varies by language:
        - 1 English character ≈ 0.3 token.
        - 1 Chinese character ≈ 0.6 token.
        -- https://api-docs.deepseek.com/quick_start/token_usage
        
        Thus: Use deepseek's tokenizer to accurately count tokens.
        '''
        self.count_tokens = tokenizer.count_tokens

        self.client = openai.AsyncOpenAI(api_key=openai.api_key or DEEPSEEK_API_KEY)
        if not self.client.api_key and check_openai_key:
            raise MissingAPIKeyException(
                'Please set the DEEPSEEK_API_KEY environment variable before using the chatgpt translator.')
        self.client.base_url = DEEPSEEK_API_BASE
        self.token_count = 0
        self.token_count_last = 0
        self.config = None

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config


    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        """
        原脚本中用来把多个 query 组装到一个 Prompt。
        同时可以做长度控制，如果过长就切分成多个 prompt。

        Original script's method to assemble multiple queries into prompts.
        Handles length control by splitting long queries into multiple prompts.

        """
        chunk_queries = []
        current_length = 0
        batch = []

        # Buffer for ID tag prepended to each query. 
        # Assume 1 token per char (worst case scenario)
        # 
        # - Use `len(queries)` to get max digit count
        #   (i.e. 0-9 => 1, 10-99 => 2, 100-999 => 3, etc.)
        IDTagBuffer=len(f"\n<|{len(queries)}|>")
        
        for q in queries:
            qTokens=self.count_tokens(q) + IDTagBuffer

            if batch and ( (current_length + qTokens) > self._MAX_TOKENS_IN):
                # 输出当前 batch
                # Output current batch
                chunk_queries.append(batch)
                batch = []
                current_length = 0
            
            batch.append(q)
            current_length += qTokens
        if batch:
            chunk_queries.append(batch)

        # 逐个批次生成 prompt
        # Generate prompts batch by batch
        for this_batch in chunk_queries:
            prompt = ""
            if self.include_template:
                prompt = self.prompt_template.format(to_lang=to_lang)
            
            # 加上分行内容
            # Add line breaks
            for i, query in enumerate(this_batch):
                prompt += f"\n<|{i+1}|>{query}"
            
            yield prompt.lstrip(), len(this_batch)


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

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:  
        translations = [''] * len(queries)  
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')  
        MAX_SPLIT_ATTEMPTS = 5  # Default max split attempts  
        RETRY_ATTEMPTS = self._RETRY_ATTEMPTS  

        async def translate_batch(prompt_queries, prompt_query_indices, split_level=0):  
            nonlocal MAX_SPLIT_ATTEMPTS
            split_prefix = ' (split)' if split_level > 0 else ''  

            # Assemble prompt for the current batch  
            prompt, query_size = self._assemble_prompts(from_lang, to_lang, prompt_queries).__next__()  
            self.logger.debug(f'-- GPT Prompt{split_prefix} --\n' + self._format_prompt_log(to_lang, prompt))  

            for attempt in range(RETRY_ATTEMPTS):  
                try:  
                    # Start the translation request with timeout handling
                    request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                    started = time.time()
                    timeout_attempt = 0
                    ratelimit_attempt = 0
                    server_error_attempt = 0
                    while not request_task.done():
                        await asyncio.sleep(0.1)
                        if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                            # Server takes too long to respond
                            if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                                raise Exception('deepseek servers did not respond quickly enough.')
                            timeout_attempt += 1
                            self.logger.warn(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                            request_task.cancel()
                            request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                            started = time.time()

                    # Get the response
                    response = await request_task  
                    self.logger.debug(f'-- GPT Response{split_prefix} --\n' + response)  

                    # Split response into translations  
                    new_translations = re.split(r'<\|\d+\|>', response)  
                    if not new_translations[0].strip():  
                        new_translations = new_translations[1:]  

                    if len(prompt_queries) == 1 and len(new_translations) == 1 and not re.match(r'^\s*<\|\d+\|>', response):  
                        self.logger.warn(f'Single query response does not contain prefix, retrying...(Attempt {attempt + 1})')  
                        continue  

                    if len(new_translations) < query_size:  
                        # Try splitting by newlines instead  
                        new_translations = re.split(r'\n', response)  

                    if len(new_translations) < query_size:  
                        remaining_attempts = RETRY_ATTEMPTS - attempt - 1  
                        self.logger.warn(f'Incomplete response, remaining {remaining_attempts} time(s) before splitting the translation.')  
                        continue  

                    # Trim excess translations and pad if necessary  
                    new_translations = new_translations[:query_size] + [''] * (query_size - len(new_translations))  
                    # Clean translations by keeping only the content before the first newline  
                    new_translations = [t.split('\n')[0].strip() for t in new_translations]  
                    # Remove any potential prefix markers  
                    new_translations = [re.sub(r'^\s*<\|\d+\|>\s*', '', t) for t in new_translations]  
                    # Check if any translations are empty  
                    if any(not t.strip() for t in new_translations):  
                        self.logger.warn(f'Empty translations detected. Resplitting the batch.') 
                        break  # Exit retry loop and trigger split logic below 

                    # Store the translations in the correct indices  
                    for idx, translation in zip(prompt_query_indices, new_translations):  
                        translations[idx] = translation  

                    # Log progress  
                    self.logger.info(f'Batch translated: {len([t for t in translations if t])}/{len(queries)} completed.')  
                    self.logger.debug(f'Completed translations: {[t if t else queries[i] for i, t in enumerate(translations)]}')        
                    return True  # Successfully translated this batch  
                    
                except openai.APIError:  
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error(
                            'Deepseek encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)
                except Exception as e:  
                    self.logger.error(f'Error during translation attempt: {e}')  
                    if attempt == RETRY_ATTEMPTS - 1:  
                        raise  
                    await asyncio.sleep(1)  

            # If retries exhausted and still not successful, proceed to split if allowed  
            if split_level < MAX_SPLIT_ATTEMPTS:  
                if split_level == 0:  
                    self.logger.warn('Retry limit reached. Starting to split the translation batch.')  
                else:  
                    self.logger.warn('Further splitting the translation batch due to persistent errors.')  
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

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        # 构建 messages
        # Build messages
        messages = [
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},
        ]

        # 如果需要先给出示例对话
        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        # 最终用户请求
        # User request
        messages.append({'role': 'user', 'content': prompt})

        kwargs = {
            'model': DEEPSEEK_MODEL,
            'messages': messages,
            
            # `max_tokens` only affects output token length. Set to max.
            'max_tokens': self._MAX_TOKENS, 
            
            'temperature': self.temperature,
            'top_p': self.top_p,
        }
        try:
            response = await self.client.beta.chat.completions.parse(**kwargs)
            
            # 添加错误处理和日志
            if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
                self.logger.warning("Response does not contain usage information")
                self.token_count_last = 0
            else:
                self.token_count += response.usage.total_tokens
                self.token_count_last = response.usage.total_tokens
            
            # 获取响应文本
            # Get the response text
            for choice in response.choices:
                if 'text' in choice:
                    return choice.text

            # 如果响应中包含推理内容，记录下来
            # Log reasoning content if available
            if hasattr(response.choices[0].message, 'reasoning_content'):
                self.logger.debug("-- GPT Reasoning --\n" +
                                response.choices[0].message.reasoning_content +
                                "\n------------------\n"
                            )
                
            self.logger.debug("-- GPT Response --\n" +
                                response.choices[0].message.content +
                                "\n------------------\n"
                            )

            # If no response with text is found, return the first response's content (which may be empty)
            # 如果没有找到包含文本的响应，则返回第一个响应的内容（可能为空）
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Error in _request_translation: {str(e)}")
            raise
