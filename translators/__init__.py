
from typing import List
from . import baidu, google, youdao, deepl, papago

import googletrans


LANGUAGE_CODE_MAP = {}

VALID_LANGUAGES = {
	"CHS": "Chinese (Simplified)",
	"CHT": "Chinese (Traditional)",
	"CSY": "Czech",
	"NLD": "Dutch",
	"ENG": "English",
	"FRA": "French",
	"DEU": "German",
	"HUN": "Hungarian",
	"ITA": "Italian",
	"JPN": "Japanese",
	"KOR": "Korean",
	"PLK": "Polish",
	"PTB": "Portuguese (Brazil)",
	"ROM": "Romanian",
	"RUS": "Russian",
	"ESP": "Spanish",
	"TRK": "Turkish",
	"VIN": "Vietnamese"
}

LANGUAGE_CODE_MAP['youdao'] = {
	'CHS': 'zh-CHS',
	'CHT': 'NONE',
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

LANGUAGE_CODE_MAP['baidu'] = {
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

LANGUAGE_CODE_MAP['google'] = {
	'CHS': 'zh-CN',
	'CHT': 'zh-TW',
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

LANGUAGE_CODE_MAP['deepl'] = {
	'CHS': 'ZH',
	'CHT': 'ZH',
	'JPN': "JA",
	'ENG': 'EN-US',
	'KOR': 'NONE',
	'VIN': 'NONE',
	'CSY': 'CS',
	'NLD': 'NL',
	'FRA': 'FR',
	'DEU': 'DE',
	'HUN': 'HU',
	'ITA': 'IT',
	'PLK': 'PL',
	'PTB': 'PT-BR',
	'ROM': 'RO',
	'RUS': 'RU',
	'ESP': 'ES',
	'TRK': 'NONE',
}

LANGUAGE_CODE_MAP['papago'] = {
	'CHS': 'zh-CN',
	'CHT': 'zh-TW',
	'JPN': "ja",
	'ENG': 'en',
	'KOR': 'ko',
	'VIN': 'vi',
	'CSY': 'NONE',
	'NLD': 'NONE',
	'FRA': 'fr',
	'DEU': 'de',
	'HUN': 'NONE',
	'ITA': 'it',
	'PLK': 'NONE',
	'PTB': 'pt',
	'ROM': 'NONE',
	'RUS': 'ru',
	'ESP': 'es',
	'TRK': 'NONE',
}

GOOGLE_CLIENT = google.Translator()
BAIDU_CLIENT = baidu.Translator()
YOUDAO_CLIENT = youdao.Translator()
PAPAGO_CLIENT = papago.Translator()
try:
	DEEPL_CLIENT = deepl.Translator()
except Exception as e:
	DEEPL_CLIENT = GOOGLE_CLIENT
	print(f'fail to initialize deepl :\n{str(e)} \nswitch to google translator')


async def dispatch(translator: str, src_lang: str, tgt_lang: str, texts: List[str], *args, **kwargs) -> List[str] :
	if translator not in ['google', 'youdao', 'baidu', 'deepl', 'eztrans', 'papago', 'null'] :
		raise Exception
	if translator == 'null' :
		return texts
	if not texts :
		return texts
	if tgt_lang not in VALID_LANGUAGES :
		raise Exception
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto' :
		raise Exception
	if translator == 'eztrans':
		tgt_lang = 'KOR'
		src_lang = 'JPN'
	else:
		tgt_lang = LANGUAGE_CODE_MAP[translator][tgt_lang]
		src_lang = LANGUAGE_CODE_MAP[translator][src_lang] if src_lang != 'auto' else 'auto'
		
	if tgt_lang == 'NONE' or src_lang == 'NONE' :
		raise Exception

	if translator == 'google' :
		concat_texts = '\n'.join(texts)
		empty_l = 0
		for txt in texts:
			if txt == '':
				empty_l += 1
			else:
				break
		result = await GOOGLE_CLIENT.translate(concat_texts, tgt_lang, src_lang, *args, **kwargs)
		if not isinstance(result, list):
			result = empty_l * [''] + result.text.split('\n')
			empty_r = len(concat_texts) - len(result)
			if empty_r > 0:
				result = result + empty_r * ['']
		result = [text.lstrip().rstrip() for text in result]

	elif translator == 'baidu' :
		concat_texts = '\n'.join(texts)
		result = await BAIDU_CLIENT.translate(src_lang, tgt_lang, concat_texts)
	elif translator == 'youdao' :
		concat_texts = '\n'.join(texts)
		result = await YOUDAO_CLIENT.translate(src_lang, tgt_lang, concat_texts)
	elif translator == 'deepl' :
		concat_texts = '\n'.join(texts)
		result = await DEEPL_CLIENT.translate(src_lang, tgt_lang, concat_texts)
	elif translator == 'papago' :
		concat_texts = '\n'.join(texts)
		result = await PAPAGO_CLIENT.translate(src_lang, tgt_lang, concat_texts)
	translated_sentences = []
	if len(result) < len(texts) :
		translated_sentences.extend(result)
		translated_sentences.extend([''] * (len(texts) - len(result)))
	elif len(result) > len(texts) :
		translated_sentences.extend(result[: len(texts)])
	else :
		translated_sentences.extend(result)
	return translated_sentences

async def test() :
	src = '测试'
	print(await dispatch('google', 'auto', 'ENG', [src]))

if __name__ == '__main__' :
	import asyncio
	asyncio.run(test())
