
# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from imp import reload

import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
from key import APP_KEY, APP_SECRET

def encrypt(signStr):
	hash_algorithm = hashlib.sha256()
	hash_algorithm.update(signStr.encode('utf-8'))
	return hash_algorithm.hexdigest()


def truncate(q):
	if q is None:
		return None
	size = len(q)
	return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
	headers = {'Content-Type': 'application/x-www-form-urlencoded'}
	return requests.post(YOUDAO_URL, data=data, headers=headers)

class Translator(object):
	def __init__(self):
	   pass

	def translate(self, from_lang, to_lang, query_text):
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

		response = do_request(data)
		contentType = response.headers['Content-Type']
		if contentType == "audio/mp3":
			return []
		else:
			result = response.json()
			result_list = []
			for ret in result["translation"]:
				result_list.extend(ret.split('\n'))
			return result_list

