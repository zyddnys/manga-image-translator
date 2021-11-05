
# -*- coding: utf-8 -*-
import uuid
import hashlib
import hmac, base64
import aiohttp
import time
from urllib.parse import quote

PAPAGO_URL = 'https://papago.naver.com/apis/n2mt/translate'
async def do_request(data):
	guid = uuid.uuid4()
	timestamp = int(time.time() * 1000)
	key = "v1.6.3_4f4591fdf3".encode("utf-8")
	code = f"{guid}\n{PAPAGO_URL}\n{timestamp}".encode("utf-8")
	token = base64.b64encode(hmac.new(key, code, "MD5").digest()).decode("utf-8")
	headers = {
		"Authorization": f"PPG {guid}:{token}",
		"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
		"Timestamp": str(timestamp),
	}
	async with aiohttp.ClientSession() as session:
		async with session.post(PAPAGO_URL, data=data, headers=headers) as resp:
			return await resp.json()

class Translator(object):
	def __init__(self):
		 pass

	async def translate(self, from_lang, to_lang, query_text):
		data = {}
		data['honorific'] = "false"
		data['source'] = from_lang
		data['target'] = to_lang
		data['text'] = query_text
		result = await do_request(data)
		result_list = [str.strip() for str in result["translatedText"].split("\n")]
		#result_list = []
		#for ret in result["translatedText"]:
		#	result_list.extend(ret.split('\n'))
		return result_list