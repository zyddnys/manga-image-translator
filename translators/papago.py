
# -*- coding: utf-8 -*-
from functools import cached_property
import uuid
import hmac, base64
import aiohttp
import time
import requests
import re

from translators.common import CommonTranslator

class PapagoTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh-CN',
        'CHT': 'zh-TW',
        'JPN': 'ja',
        'ENG': 'en',
        'KOR': 'ko',
        'VIN': 'vi',
        'FRA': 'fr',
        'DEU': 'de',
        'ITA': 'it',
        'PTB': 'pt',
        'RUS': 'ru',
        'ESP': 'es',
    }
    _API_URL = 'https://papago.naver.com/apis/n2mt/translate'

    async def _translate(self, from_lang, to_lang, queries):
        data = {}
        data['honorific'] = "false"
        data['source'] = from_lang
        data['target'] = to_lang
        data['text'] = '\n'.join(queries)
        result = await self._do_request(data, self._version_key)
        result_list = [str.strip() for str in result["translatedText"].split("\n")]
        return result_list

    @cached_property
    def _version_key(self):
        script = requests.get('https://papago.naver.com')
        mainJs = re.search(r'\/(main.*\.js)', script.text).group(1)
        papagoVerData = requests.get('https://papago.naver.com/' + mainJs)
        papagoVer = re.search(r'"PPG .*,"(v[^"]*)', papagoVerData.text).group(1)
        return papagoVer

    async def _do_request(self, data, version_key):
        guid = uuid.uuid4()
        timestamp = int(time.time() * 1000)
        key = version_key.encode("utf-8")
        code = f"{guid}\n{self._API_URL}\n{timestamp}".encode("utf-8")
        token = base64.b64encode(hmac.new(key, code, "MD5").digest()).decode("utf-8")
        headers = {
            "Authorization": f"PPG {guid}:{token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Timestamp": str(timestamp),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self._API_URL, data=data, headers=headers) as resp:
                return await resp.json()
