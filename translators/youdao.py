
# -*- coding: utf-8 -*-
import uuid
import hashlib
import time

import aiohttp
import time

YOUDAO_URL = 'https://openapi.youdao.com/api'
from .keys import APP_KEY, APP_SECRET

def encrypt(signStr):
	hash_algorithm = hashlib.sha256()
	hash_algorithm.update(signStr.encode('utf-8'))
	return hash_algorithm.hexdigest()


def truncate(q):
	if q is None:
		return None
	size = len(q)
	return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


async def do_request(data):
	headers = {'Content-Type': 'application/x-www-form-urlencoded'}
	async with aiohttp.ClientSession() as session:
		async with session.post(YOUDAO_URL, data=data, headers=headers) as resp:
			return await resp.json()

class Translator(object):
	def __init__(self):
		pass

	async def translate(self, from_lang, to_lang, query_text):
		data = {}
		data['from'] = from_lang
		data['to'] = to_lang
		data['signType'] = 'v3'
		curtime = str(int(time.time()))
		data['curtime'] = curtime
		salt = str(uuid.uuid1())
		signStr = APP_KEY + truncate(query_text) + salt + curtime + APP_SECRET
		sign = encrypt(signStr)
		data['appKey'] = APP_KEY
		data['q'] = query_text
		data['salt'] = salt
		data['sign'] = sign
		#data['vocabId'] = "您的用户词表ID"

		result = await do_request(data)
		result_list = []
		for ret in result["translation"]:
			result_list.extend(ret.split('\n'))
		return result_list

