
from typing import List
import asyncio

from translators.common import CommonTranslator
from .baidu import BaiduTranslator
from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .offline import OfflineTranslator

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

translators = {}

def get_translator(key: str, *args, **kwargs) -> CommonTranslator:
	def set_and_return(key, translator):
		if key not in translators:
			translators[key] = translator(*args, **kwargs)
		return translators[key]

	if key == 'google':
		return set_and_return(key, GoogleTranslator)
	elif key == 'youdao':
		return set_and_return(key, BaiduTranslator)
	elif key == 'baidu':
		return set_and_return(key, YoudaoTranslator)
	elif key == 'deepl':
		try:
			return set_and_return(key, DeeplTranslator)
		except Exception as e:
			print(f'failed to initialize deepl :\n{str(e)}\nswitching to google translator')
			return set_and_return(key, GoogleTranslator)
	elif key == 'papago':
		return set_and_return(key, PapagoTranslator)
	elif key == 'offline' or key == 'offline_big':
		translator = set_and_return(key, OfflineTranslator)
		return translator

async def dispatch(translator_key: str, src_lang: str, tgt_lang: str, queries: List[str], *args, **kwargs) -> List[str] :
	if translator_key not in ['google', 'youdao', 'baidu', 'deepl', 'eztrans', 'papago', 'offline', 'offline_big', 'null'] :
		raise Exception
	if translator_key == 'null' :
		return queries
	if not queries :
		return queries

	if translator_key == 'eztrans':
		tgt_lang = 'KOR'
		src_lang = 'JPN'

	if tgt_lang not in VALID_LANGUAGES :
		raise Exception('Invalid language code: "%s", please choose from the following: %s' % (tgt_lang, ','.join(VALID_LANGUAGES)))
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto' :
		raise Exception('Invalid language code: "%s", please choose from the following: auto,%s' % (src_lang, ','.join(VALID_LANGUAGES)))
	
	translator = get_translator(translator_key)

	if translator_key in ('offline', 'offline_big'):
		if not translator.is_loaded():
			translator.load(translator_key)
		result = await asyncio.create_task(translator.translate(src_lang, tgt_lang, queries))
	else:
		result = await translator.translate(src_lang, tgt_lang, queries)
		
	translated_sentences = []
	if len(result) < len(queries) :
		translated_sentences.extend(result)
		translated_sentences.extend([''] * (len(queries) - len(result)))
	elif len(result) > len(queries) :
		translated_sentences.extend(result[: len(queries)])
	else :
		translated_sentences.extend(result)
	return translated_sentences

def test() :
	src = '测试'
	print(dispatch('offline', 'auto', 'ENG', [src]))

if __name__ == '__main__' :
	import asyncio
	asyncio.run(test())
