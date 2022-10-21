import asyncio
from typing import List

from .common import CommonTranslator
from .baidu import BaiduTranslator
from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .offline import OfflineTranslator

VALID_LANGUAGES = {
	'CHS': 'Chinese (Simplified)',
	'CHT': 'Chinese (Traditional)',
	'CSY': 'Czech',
	'NLD': 'Dutch',
	'ENG': 'English',
	'FRA': 'French',
	'DEU': 'German',
	'HUN': 'Hungarian',
	'ITA': 'Italian',
	'JPN': 'Japanese',
	'KOR': 'Korean',
	'PLK': 'Polish',
	'PTB': 'Portuguese (Brazil)',
	'ROM': 'Romanian',
	'RUS': 'Russian',
	'ESP': 'Spanish',
	'TRK': 'Turkish',
	'VIN': 'Vietnamese',
}

OFFLINE_TRANSLATORS = {
	'offline': OfflineTranslator,
	'offline_big': OfflineTranslator,
}

TRANSLATORS = {
	'google': GoogleTranslator,
	'youdao': YoudaoTranslator,
	'baidu': BaiduTranslator,
	'deepl': DeeplTranslator,
	'papago': PapagoTranslator,
	**OFFLINE_TRANSLATORS,
}
translator_cache = {}

def get_translator(key: str, *args, **kwargs) -> CommonTranslator:
	if key not in TRANSLATORS:
		raise Exception(f'Could not find translator for: "{key}". Choose from the following: %s' % ', '.join(TRANSLATORS))
	if key not in translator_cache:
		translator = TRANSLATORS[key]
		translator_cache[key] = translator(*args, **kwargs)
	return translator_cache[key]

async def dispatch(translator_key: str, src_lang: str, tgt_lang: str, queries: List[str], **kwargs) -> List[str]:
	if translator_key == 'null':
		return queries
	if not queries:
		return queries

	if tgt_lang not in VALID_LANGUAGES:
		raise Exception('Invalid language code: "%s". Choose from the following: %s' % (tgt_lang, ', '.join(VALID_LANGUAGES)))
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto':
		raise Exception('Invalid language code: "%s". Choose from the following: auto, %s' % (src_lang, ', '.join(VALID_LANGUAGES)))
	
	# Might want to remove this fallback in the future, as its misleading
	if translator_key == 'deepl':
		try:
			translator = get_translator(translator_key)
		except Exception as e:
			print(f'Failed to initialize deepl :\n{str(e)}\nFallback to google translator')
			translator = get_translator('google')
	else:
		translator = get_translator(translator_key)

	if translator_key in ('offline', 'offline_big'):
		if not translator.is_loaded():
			translator.load(translator_key, kwargs.get('use_cuda', False))
		result = await asyncio.create_task(translator.translate(src_lang, tgt_lang, queries))
	else:
		result = await translator.translate(src_lang, tgt_lang, queries)
		
	translated_sentences = []
	if len(result) < len(queries):
		translated_sentences.extend(result)
		translated_sentences.extend([''] * (len(queries) - len(result)))
	elif len(result) > len(queries):
		translated_sentences.extend(result[:len(queries)])
	else:
		translated_sentences.extend(result)
	return translated_sentences

def test():
	src = '测试'
	print(dispatch('offline', 'auto', 'ENG', [src]))

if __name__ == '__main__':
	import asyncio
	asyncio.run(test())
