
# -*- coding: utf-8 -*-
import aiohttp

from .common import CommonTranslator, InvalidServerResponse, MissingAPIKeyException
from .keys import CAIYUN_TOKEN

class CaiyunTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh',
        'CHT': 'zh-Hant',
        'ENG': 'en',
        'JPN': 'ja',
        'KOR': 'ko',
        'DEU': 'de',
        'ESP': 'es',
        'FRA': 'fr',
        'ITA': 'it',
        'PTB': 'pt',
        'RUS': 'ru',
        'TUR': 'tr',
        'VIN': 'vi',
    }
    _API_URL = 'https://api.interpreter.caiyunai.com/v1/translator'

    def __init__(self):
        super().__init__()
        if not CAIYUN_TOKEN:
            raise MissingAPIKeyException('Please set the CAIYUN_TOKEN environment variables before using the caiyun translator.')

    async def _translate(self, from_lang, to_lang, queries):
        data = {
            "trans_type": from_lang + "2" + to_lang,
            "source": queries,
            "request_id": "manga-image-translator"
        }
        if from_lang == "auto":
            data["detect"] = True

        result = await self._do_request(data)
        if "target" not in result:
            raise InvalidServerResponse(f'Caiyun returned invalid response: {result}\nAre the API keys set correctly?')
        return result["target"]

    def _truncate(self, q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    async def _do_request(self, data):
        headers = {
            "content-type": "application/json",
            "x-authorization": "token " + CAIYUN_TOKEN,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self._API_URL, json=data, headers=headers) as resp:
                return await resp.json()
