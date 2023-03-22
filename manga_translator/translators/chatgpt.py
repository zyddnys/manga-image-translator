import re
import openai
import asyncio
import time

from .common import CommonTranslator, MissingAPIKeyException
from .keys import OPENAI_API_KEY

# Example query:
"""Please help me to translate the following queries to english:
Query 1:
ちょっと悪いんだけど

Query 2:
そこの魔法陣に入って頂戴

Query 3:
いやいや何も起きないから(嘘)
是否离线可用

Query 4:
いやいや何も起きないから(嘘)
是否离线可用

Translation 1:

"""

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

    def __init__(self):
        super().__init__()
        openai.api_key = openai.api_key or OPENAI_API_KEY
        if not openai.api_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')

    async def _translate(self, from_lang, to_lang, queries):
        prompt = f'Please help me to translate the following text from a manga to {to_lang}:\n'
        for i, query in enumerate(queries):
            prompt += f'\nText {i+1}:\n{query}\n'
        prompt += '\nTranslation 1:\n'
        self.logger.debug(prompt)

        request_task = asyncio.create_task(self._request_translation(prompt))
        started = time.time()
        attempts = 0
        while not request_task.done():
            await asyncio.sleep(0.1)
            if time.time() - started > 20:
                if attempts >= 3:
                    raise Exception('API servers did not respond quickly enough.')
                self.logger.info(f'Restarting request due to timeout. Attempt: {attempts+1}')
                request_task.cancel()
                request_task = asyncio.create_task(self._request_translation(prompt))
                started = time.time()
                attempts += 1
        response = await request_task

        self.logger.debug(response)
        translations = re.split(r'Translation \d+:\n', response)
        translations = [t.strip() for t in translations]
        self.logger.debug(translations)
        return translations

    async def _request_translation(self, prompt: str) -> str:
        completion = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=1024,
            temperature=1,
        )
        response = completion.choices[0].text
        return response
