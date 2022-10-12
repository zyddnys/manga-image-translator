
#from https://pypi.org/project/baidu-trans/

import requests
import hashlib
import urllib.parse
import random
import traceback
import json
import time

from translators.common import CommonTranslator

from .keys import APP_ID, SECRET_KEY

# base api url
BASE_URL = 'api.fanyi.baidu.com'
API_URL = '/api/trans/vip/translate'

LANGUAGE_CODE_MAP = {
	'CHS': 'zh',
	'CHT': 'cht',
	'JPN': "ja",
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
	'TRK': 'NONE',
}

import aiohttp

# FIXME: Baidu translator api outdated
class BaiduTranslator(CommonTranslator):
	def __init__(self):
		pass

	def _get_language_code(self, key):
		return LANGUAGE_CODE_MAP[key]

	async def _translate(self, from_lang, to_lang, queries):
		url = self.get_url(from_lang, to_lang, '\n'.join(queries))
		async with aiohttp.ClientSession() as session:
			async with session.get('https://'+BASE_URL+url) as resp:
				result = await resp.json()
		result_list = []
		for ret in result["trans_result"]:
			for v in ret["dst"].split('\n') :
				result_list.append(v)
		return result_list

	@staticmethod
	def get_url(from_lang, to_lang, query_text):
		# 随机数据
		salt = random.randint(32768, 65536)
		# MD5生成签名
		sign = APP_ID + query_text + str(salt) + SECRET_KEY
		m1 = hashlib.md5()
		m1.update(sign.encode('utf-8'))
		sign = m1.hexdigest()
		# 拼接URL
		url = API_URL +'?appid=' + APP_ID + '&q=' + urllib.parse.quote(query_text) + '&from=' + from_lang + '&to=' + to_lang + '&salt=' + str(salt) + '&sign=' + sign
		return url

