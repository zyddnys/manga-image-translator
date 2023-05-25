import re
import openai
import openai.error
import asyncio
import time
from typing import List, Literal

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY

SIMPLE_PROMPT_TEMPLATE = 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n'
PROMPT_OVERWRITE = None
TEMPERATURE_OVERWRITE = 0.5

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
    }
    _INVALID_REPEAT_COUNT = 2 # repeat 2 times at most if invalid translation was returned
    _MAX_REQUESTS_PER_MINUTE = 20
    _RETRY_ATTEMPTS = 3

    _MAX_TOKENS = 4096
    _prompt_template = SIMPLE_PROMPT_TEMPLATE

    @property
    def prompt_template(self):
        global PROMPT_OVERWRITE
        return PROMPT_OVERWRITE or self._prompt_template

    @property
    def temperature(self):
        global TEMPERATURE_OVERWRITE
        return TEMPERATURE_OVERWRITE

    def __init__(self):
        super().__init__()
        openai.api_key = openai.api_key or OPENAI_API_KEY
        if not openai.api_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')
        if OPENAI_HTTP_PROXY:
            proxies = {
                'http': 'http://%s' % OPENAI_HTTP_PROXY,
                'https': 'http://%s' % OPENAI_HTTP_PROXY
            }
            openai.proxy = proxies
        self.token_count = 0
        self.token_count_last = 0

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        prompt = self.prompt_template.format(to_lang=to_lang)
        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\nText {i+1-i_offset}:\n{query}\n'
            # If prompt is growing too large and theres still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i+1:])) > self._MAX_TOKENS:
                prompt += '\nTranslation 1:\n'
                yield prompt
                prompt = self.prompt_template.format(to_lang=to_lang)
                # Restart counting at 1
                i_offset = i + 1

        prompt += '\nTranslation 1:\n'
        yield prompt

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}')

        for prompt in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- GPT Prompt --\n' + prompt)

            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            while True:
                request_task = asyncio.create_task(self._request_translation(prompt))
                started = time.time()
                while not request_task.done():
                    await asyncio.sleep(0.1)
                    if time.time() - started > 30: # Server takes too long to respond
                        if timeout_attempt >= 3:
                            raise Exception('openai servers did not respond quickly enough.')
                        timeout_attempt += 1
                        self.logger.warn(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                        request_task.cancel()
                        request_task = asyncio.create_task(self._request_translation(prompt))
                        started = time.time()
                try:
                    response = await request_task
                    break
                except openai.error.RateLimitError: # Server returned ratelimit response
                    ratelimit_attempt += 1
                    if ratelimit_attempt >= 3:
                        raise
                    self.logger.warn(f'Restarting request due to ratelimiting by openai servers. Attempt: {ratelimit_attempt}')
                    await asyncio.sleep(2)
                except openai.error.APIError: # Server returned 500 error (probably server load)
                    server_error_attempt += 1
                    if server_error_attempt >= self._RETRY_ATTEMPTS:
                        self.logger.error('OpenAI encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                        raise
                    self.logger.warn(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                    await asyncio.sleep(1)

            self.logger.debug('-- GPT Response --\n' + response)
            new_translations = re.split(r'Translation \d+:\n', response)
            translations.extend([t.strip() for t in new_translations])

        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    async def _request_translation(self, prompt: str) -> str:
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=2048,
            temperature=self.temperature,
        )
        self.token_count += response.usage['total_tokens']
        self.token_count_last = response.usage['total_tokens']
        return response.choices[0].text

class GPT35TurboTranslator(GPT3Translator):
    _MAX_REQUESTS_PER_MINUTE = 200
    PROMPT_TEMPLATE = SIMPLE_PROMPT_TEMPLATE

    async def _request_translation(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': 'You are a professional translator who will follow the required format for translation.'},
            {'role': 'user', 'content': prompt},
        ]

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=2048,
            temperature=self.temperature,
        )

        self.token_count += response.usage['total_tokens']
        self.token_count_last = response.usage['total_tokens']
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

class GPT4Translator(GPT3Translator):
    _MAX_REQUESTS_PER_MINUTE = 200
    PROMPT_TEMPLATE = SIMPLE_PROMPT_TEMPLATE
    _RETRY_ATTEMPTS = 5

    async def _request_translation(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': 'You are a professional translator who will follow the required format for translation.'},
            {'role': 'user', 'content': prompt},
        ]

        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            max_tokens=4096,
            temperature=self.temperature,
        )

        self.token_count += response.usage['total_tokens']
        self.token_count_last = response.usage['total_tokens']
        for choice in response.choices:
            if 'text' in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content
