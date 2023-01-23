
#from https://pypi.org/project/baidu-trans/

import hashlib
import urllib.parse
import random

from translators.common import CommonTranslator
from .keys import BAIDU_APP_ID, BAIDU_SECRET_KEY

# base api url
BASE_URL = 'api.fanyi.baidu.com'
API_URL = '/api/trans/vip/translate'

import aiohttp

class BaiduTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh',
        'CHT': 'cht',
        'JPN': 'ja',
        'ENG': 'en',
        'KOR': 'kor',
        'VIN': 'vie',
        'CSY': 'cs',
        'NLD': 'nl',
        'FRA': 'fra',
        'DEU': 'de',
        'HUN': 'hu',
        'ITA': 'it',
        'PLK': 'pl',
        'PTB': 'pt',
        'ROM': 'rom',
        'RUS': 'ru',
        'ESP': 'spa',
    }

    def __init__(self) -> None:
        super().__init__()
        if not BAIDU_APP_ID or not BAIDU_SECRET_KEY:
            raise ValueError('Please set the BAIDU_APP_ID and BAIDU_SECRET_KEY environment variables before using the baidu translator.')

    async def _translate(self, from_lang, to_lang, queries):
        url = self.get_url(from_lang, to_lang, '\n'.join(queries))
        async with aiohttp.ClientSession() as session:
            async with session.get('https://'+BASE_URL+url) as resp:
                result = await resp.json()
        result_list = []
        for ret in result["trans_result"]:
            for v in ret["dst"].split('\n'):
                result_list.append(v)
        return result_list

    @staticmethod
    def get_url(from_lang, to_lang, query_text):
        # 随机数据
        salt = random.randint(32768, 65536)
        # MD5生成签名
        sign = BAIDU_APP_ID + query_text + str(salt) + BAIDU_SECRET_KEY
        m1 = hashlib.md5()
        m1.update(sign.encode('utf-8'))
        sign = m1.hexdigest()
        # 拼接URL
        url = API_URL +'?appid=' + BAIDU_APP_ID + '&q=' + urllib.parse.quote(query_text) + '&from=' + from_lang + '&to=' + to_lang + '&salt=' + str(salt) + '&sign=' + sign
        return url

