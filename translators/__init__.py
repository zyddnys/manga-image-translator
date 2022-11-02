import asyncio
from typing import List

from .common import CommonTranslator, OfflineTranslator
from .baidu import BaiduTranslator
from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .nnlb import NNLBTranslator, NNLBBigTranslator
from .jparacrawl import JParaCrawlTranslator, JParaCrawlSmallTranslator, JParaCrawlBigTranslator
from .no import NoTranslator

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

# TODO: Implement automatic offline translator assignment through a special class
OFFLINE_TRANSLATORS = {
	# 'offline': None,
	# 'offline_big': None,
	'nnlb': NNLBTranslator,
	'nnlb_big': NNLBBigTranslator,
	'sugoi': JParaCrawlTranslator,
	'sugoi_small': JParaCrawlSmallTranslator,
	'sugoi_big': JParaCrawlBigTranslator,
}

TRANSLATORS = {
	'google': GoogleTranslator,
	'youdao': YoudaoTranslator,
	'baidu': BaiduTranslator,
	'deepl': DeeplTranslator,
	'papago': PapagoTranslator,
	'no': NoTranslator,
	**OFFLINE_TRANSLATORS,
}
translator_cache = {}


def get_suitable_offline_translator_key(src_lang: str, tgt_lang: str, big_model: bool = False) -> str:
	if src_lang == 'JPN' or tgt_lang == 'JPN':
		return 'sugoi_big' if big_model else 'sugoi'
	return 'nnlb_big' if big_model else 'nnlb'

def get_translator(key: str, src_lang: str = None, tgt_lang: str = None, *args, **kwargs) -> CommonTranslator:
	if key not in TRANSLATORS:
		raise ValueError(f'Could not find translator for: "{key}". Choose from the following: %s' % ', '.join(TRANSLATORS))
	if key == 'offline' or key == 'offline_big':
		if not src_lang or not tgt_lang:
			raise Exception(f'Translator key: "{key}" required src_lang and tgt_lang to be set.')
		key = get_suitable_offline_translator_key(src_lang, tgt_lang, key == 'offline_big')
	if not translator_cache.get(key):
		translator = TRANSLATORS[key]
		translator_cache[key] = translator(*args, **kwargs)
	return translator_cache[key]

async def prepare(translator_key: str, src_lang: str, tgt_lang: str):
	translator = get_translator(translator_key, src_lang, tgt_lang)
	if src_lang not in translator.supported_src_languages:
		raise ValueError(f'Translator "{translator_key}" does not support language "{src_lang}". ' +
						 f'Please choose from: {",".join(translator.supported_src_languages)}.')
	if tgt_lang not in translator.supported_tgt_languages:
		raise ValueError(f'Translator "{translator_key}" does not support language "{tgt_lang}". ' +
						 f'Please choose from: {",".join(translator.supported_tgt_languages)}.')
	if isinstance(translator, OfflineTranslator):
		await translator.download()

async def dispatch(translator_key: str, src_lang: str, tgt_lang: str, queries: List[str], **kwargs) -> List[str]:
	if translator_key == 'null':
		return queries
	if not queries:
		return queries

	if tgt_lang not in VALID_LANGUAGES:
		raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (tgt_lang, ', '.join(VALID_LANGUAGES)))
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto':
		raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (src_lang, ', '.join(VALID_LANGUAGES)))

	# Might want to remove this fallback in the future, as its misleading
	if translator_key == 'deepl':
		try:
			translator = get_translator(translator_key)
		except Exception as e:
			print(f'Failed to initialize deepl: {str(e)}. Fallback to google translator')
			translator = get_translator('google')
	else:
		translator = get_translator(translator_key, src_lang, tgt_lang)

	if isinstance(translator, OfflineTranslator):
		if not translator.is_loaded():
			device = 'cuda' if kwargs.get('use_cuda', False) else 'cpu'
			await translator.load(src_lang, tgt_lang, device)
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

async def test():
	src = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
	translator = 'sugoi_small'
	await prepare(translator, 'auto', 'ENG')
	print(await dispatch(translator, 'auto', 'ENG', src))
