
# -*- coding: utf-8 -*-
import uuid
import hashlib
import time
import aiohttp
import time

from translators.common import CommonTranslator
from .keys import YOUDAO_APP_KEY, YOUDAO_SECRET_KEY

def sha256_encode(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

class YoudaoTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh-CHS',
        'JPN': "ja",
        'ENG': 'en',
        'KOR': 'ko',
        'VIN': 'vi',
        'CSY': 'cs',
        'NLD': 'nl',
        'FRA': 'fr',
        'DEU': 'de',
        'HUN': 'hu',
        'ITA': 'it',
        'PLK': 'pl',
        'PTB': 'pt',
        'ROM': 'ro',
        'RUS': 'ru',
        'ESP': 'es',
        'TRK': 'tr',
    }
    _API_URL = 'https://openapi.youdao.com/api'

    def __init__(self) -> None:
        super().__init__()
        if not YOUDAO_APP_KEY or not YOUDAO_SECRET_KEY:
            raise ValueError('Please set the YOUDAO_APP_KEY and YOUDAO_SECRET_KEY environment variables before using the youdao translator.')

    async def _translate(self, from_lang, to_lang, queries):
        data = {}
        query_text = '\n'.join(queries)
        data['from'] = from_lang
        data['to'] = to_lang
        data['signType'] = 'v3'
        curtime = str(int(time.time()))
        data['curtime'] = curtime
        salt = str(uuid.uuid1())
        signStr = YOUDAO_APP_KEY + self._truncate(query_text) + salt + curtime + YOUDAO_SECRET_KEY
        sign = sha256_encode(signStr)
        data['appKey'] = YOUDAO_APP_KEY
        data['q'] = query_text
        data['salt'] = salt
        data['sign'] = sign
        #data['vocabId'] = "您的用户词表ID"

        result = await self._do_request(data)
        result_list = []
        for ret in result["translation"]:
            result_list.extend(ret.split('\n'))
        return result_list

    def _truncate(self, q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    async def _do_request(self, data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        async with aiohttp.ClientSession() as session:
            async with session.post(self._API_URL, data=data, headers=headers) as resp:
                return await resp.json()
