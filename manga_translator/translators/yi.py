import re
import asyncio
import time
from typing import List, Dict
import requests
import json

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY, OPENAI_HTTP_PROXY, OPENAI_API_BASE

CONFIG = None

class YITranslator(CommonTranslator):
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
    _PROMPT_TEMPLATE = 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n'

    kconfig = None

    def __init__(self):
        super().__init__()

        self.token_count = 0
        self.token_count_last = 0
        self.config = None

    def parse_args(self, args):
        self.config = args.gpt_config
        self.kconfig = args

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
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, prompt))

            ratelimit_attempt = 0
            server_error_attempt = 0
            timeout_attempt = 0
            response = self._request_translation(to_lang, prompt)
            self.logger.debug('-- GPT Response --\n' + response)

            new_translations = re.split(r'<\|\d+\|>', response)
            # When there is only one query chatgpt likes to exclude the <|1|>
            if not new_translations[0].strip():
                new_translations = new_translations[1:]

            if len(new_translations) <= 1 and query_size > 1:
                # Try splitting by newlines instead
                new_translations = re.split(r'\n', response)

            if len(new_translations) != query_size:
                # super method will repeat translation as per self._INVALID_REPEAT_COUNT
                translations = []
                break

            translations.extend([t.strip() for t in new_translations])

        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def _request_translation(self, to_lang: str, prompt: str) -> str:
        #response = await openai.Completion.acreate(model='text-davinci-003',prompt=prompt,max_tokens=self._MAX_TOKENStemperature=self.temperature,top_p=self.top_p,)
        #self.token_count += response.usage['total_tokens']
        #self.token_count_last = response.usage['total_tokens']
        self.logger.info(f'Prompt: {prompt}')
        return "response.choices[0].text"

class YI34bTranslator(YITranslator):
    _CONFIG_KEY = 'gpt35'
    _MAX_REQUESTS_PER_MINUTE = 200
    _RETURN_PROMPT = False
    _INCLUDE_TEMPLATE = False

    # Token: 57+
    _CHAT_SYSTEM_TEMPLATE = (
        'You are a professional translation engine, '
        'please translate the story into a colloquial, '
        'elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, '
        'never interpret it. '
        'If there is any issue in the text, '
        'output it as is.\n'
        'Translate to {to_lang}.'
    )
    _CHAT_SAMPLE = {
        'Simplified Chinese': [ # Token: 88 + 84
            (
                '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n'
                '<|2|>きみ… 大丈夫⁉\n'
                '<|3|>なんだこいつ 空気読めて ないのか…？'
            ),
            (
                '<|1|>好尴尬…我不想引人注目…我想消失…\n'
                '<|2|>你…没事吧⁉\n'
                '<|3|>这家伙怎么看不懂气氛的…？'
            ),
        ]
    }

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

    def _request_translation(self, to_lang: str, prompt: str) -> str:
        print("___________________________________________________________________")
        messages = [
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)},
            {'role': 'user', 'content': prompt},
        ]

        if to_lang in self.chat_sample:
            messages.insert(1, {'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.insert(2, {'role': 'assistant', 'content': self.chat_sample[to_lang][1]})


        self.logger.info(f'Prompt: {messages}')
        # Convert the data to JSON format
        json_data = json.dumps(messages)

        url = self.kconfig.get('PrivateGPTAddress')
        response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(response.text)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return response.text
