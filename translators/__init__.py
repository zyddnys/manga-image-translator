
from typing import List
from . import baidu, google, youdao

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

GOOGLE_CLIENT = google.Translator()
BAIDU_CLIENT = baidu.Translator()
YOUDAO_CLIENT = youdao.Translator()

async def dispatch(translator: str, src_lang: str, tgt_lang: str, texts: List[str], *args, **kwargs) -> List[str] :
	if translator not in ['google', 'youdao', 'baidu', 'null'] :
		raise Exception
	if translator == 'null' :
		return texts
	if tgt_lang not in VALID_LANGUAGES :
		raise Exception
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto' :
		raise Exception
	tgt_lang = LANGUAGE_CODE_MAP[translator][tgt_lang]
	src_lang = LANGUAGE_CODE_MAP[translator][src_lang] if src_lang != 'auto' else 'auto'
	if tgt_lang == 'NONE' or src_lang == 'NONE' :
		raise Exception
	if translator == 'google' :
		concat_texts = '\n'.join(texts)
		result = await GOOGLE_CLIENT.translate(concat_texts, tgt_lang, src_lang, *args, **kwargs)
		if not isinstance(result, list) :
			result = result.text.split('\n')
	elif translator == 'baidu' :
		concat_texts = '\n'.join(texts)
		result = await BAIDU_CLIENT.translate(src_lang, tgt_lang, concat_texts)
	elif translator == 'youdao' :
		concat_texts = '\n'.join(texts)
		result = await YOUDAO_CLIENT.translate(src_lang, tgt_lang, concat_texts)
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
