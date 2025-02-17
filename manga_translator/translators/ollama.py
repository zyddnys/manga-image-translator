import re

from ..config import TranslatorConfig
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List, Dict
from omegaconf import OmegaConf
from .common import CommonTranslator, MissingAPIKeyException, VALID_LANGUAGES
from .keys import OLLAMA_API_KEY, OLLAMA_API_BASE, OLLAMA_MODEL, OLLAMA_MODEL_CONF


class OllamaTranslator(ConfigGPT, CommonTranslator):
    _LANGUAGE_CODE_MAP=VALID_LANGUAGES

    _INVALID_REPEAT_COUNT = 2  # 如果检测到“无效”翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 4096

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False
    
    def __init__(self, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Use the `ollama` defaults.
        _CONFIG_KEY='ollama'
        if OLLAMA_MODEL_CONF:
            _CONFIG_KEY+=f".{OLLAMA_MODEL_CONF}" 
        
        ConfigGPT.__init__(self, config_key=_CONFIG_KEY) 
        CommonTranslator.__init__(self)

        self.client = openai.AsyncOpenAI(api_key=OLLAMA_API_KEY or "ollama") # required, but unused for ollama
        self.client.base_url = OLLAMA_API_BASE
        self.token_count = 0
        self.token_count_last = 0

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config


    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ''

        if self._INCLUDE_TEMPLATE:
            prompt += self.prompt_template.format(to_lang=to_lang)

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i + 1 - i_offset}|>{query}'

            # If prompt is growing too large and there's still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i + 1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                yield prompt.lstrip(), i + 1 - i_offset
                prompt = self.prompt_template.format(to_lang=to_lang)
                # Restart counting at 1
                i_offset = i + 1

        if self._RETURN_PROMPT:
            prompt += '\n<|1|>'

        yield prompt.lstrip(), len(queries) - i_offset

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
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, prompt))

            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            while True:
                request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                started = time.time()
                while not request_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                        # Server takes too long to respond
                        if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                            raise Exception('ollama servers did not respond quickly enough.')
                        timeout_attempt += 1
                        self.logger.warn(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                        request_task.cancel()
                        request_task = asyncio.create_task(self._request_translation(to_lang, prompt))
                        started = time.time()
                try:
                    response = await request_task
                    break
                except openai.RateLimitError:  # Server returned ratelimit response
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                        raise
                    self.logger.warn(
                        f'Restarting request due to ratelimiting by Ollama servers. Attempt: {ratelimit_attempt}')
                    await asyncio.sleep(2)
                except openai.APIError:  # Server returned 500 error (probably server load)
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error(
                            'Ollama encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)

            # self.logger.debug('-- GPT Response --\n' + response)
            

            # Use regex to extract response 
            response=self.extract_capture_groups(response, rf"{self.rgx_capture}")


            # Sometimes it will return line like "<|9>demo", and we need to fix it.
            def add_pipe(match):
                number = match.group(1)
                return f"<|{number}|>"
            response = re.sub(r"<\|?(\d+)\|?>", add_pipe, response)
            

            # self.logger.debug('-- GPT Response (filtered) --\n' + response)

            # @NOTE: This should *should* be superflous now, due to `extract_capture_groups`:
            # 
            # Remove any text preceeding the first translation.
            new_translations = re.split(r'<\|\d+\|>', 'pre_1\n' + response)[1:]
            # new_translations = re.split(r'<\|\d+\|>', response)

            # When there is only one query LLMs likes to exclude the <|1|>
            if not new_translations[0].strip():
                new_translations = new_translations[1:]

            if len(new_translations) <= 1 and query_size > 1:
                # Try splitting by newlines instead
                new_translations = re.split(r'\n', response)

            if len(new_translations) > query_size:
                new_translations = new_translations[: query_size]
            elif len(new_translations) < query_size:
                new_translations = new_translations + [''] * (query_size - len(new_translations))

            translations.extend([t.strip() for t in new_translations])

        for t in translations:
            if "I'm sorry, but I can't assist with that request" in t:
                raise Exception('translations contain error text')
        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        messages.append({'role': 'user', 'content': prompt})

        response = await self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        self.logger.debug('\n-- GPT Response (raw) --')
        self.logger.debug(response.choices[0].message.content)
        self.logger.debug('------------------------\n')


        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        return response.choices[0].message.content
