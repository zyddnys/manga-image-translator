import asyncio
from typing import List

from .common import CommonTranslator, OfflineTranslator
from .baidu import BaiduTranslator
from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .nllb import NLLBTranslator, NLLBBigTranslator
from .sugoi import SugoiTranslator, SugoiSmallTranslator, SugoiBigTranslator
from .selective import SelectiveOfflineTranslator, SelectiveBigOfflineTranslator, prepare as prepare_selective_translator
from .none import NoneTranslator
from .original import OriginalTranslator

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
	'UKR': 'Ukrainian',
	'VIN': 'Vietnamese',
}

OFFLINE_TRANSLATORS = {
	'offline': SelectiveOfflineTranslator,
	'offline_big': SelectiveBigOfflineTranslator,
	'nllb': NLLBTranslator,
	'nllb_big': NLLBBigTranslator,
	'sugoi': SugoiTranslator,
	'sugoi_small': SugoiSmallTranslator,
	'sugoi_big': SugoiBigTranslator,
}

TRANSLATORS = {
	'google': GoogleTranslator,
	'youdao': YoudaoTranslator,
	'baidu': BaiduTranslator,
	'deepl': DeeplTranslator,
	'papago': PapagoTranslator,
	'none': NoneTranslator,
	'original': OriginalTranslator,
	**OFFLINE_TRANSLATORS,
}
translator_cache = {}

def get_translator(key: str, *args, **kwargs) -> CommonTranslator:
	if key not in TRANSLATORS:
		raise ValueError(f'Could not find translator for: "{key}". Choose from the following: %s' % ','.join(TRANSLATORS))
	if not translator_cache.get(key):
		translator = TRANSLATORS[key]
		translator_cache[key] = translator(*args, **kwargs)
	return translator_cache[key]

prepare_selective_translator(get_translator)

async def prepare(translator_key: str, src_lang: str, tgt_lang: str):
	translator = get_translator(translator_key)
	translator.supports_languages(src_lang, tgt_lang, fatal=True)
	if isinstance(translator, OfflineTranslator):
		await translator.download()

async def dispatch(translator_key: str, src_lang: str, tgt_lang: str, queries: List[str], use_cuda: bool = False) -> List[str]:
	if not queries:
		return queries

	if tgt_lang not in VALID_LANGUAGES:
		raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (tgt_lang, ', '.join(VALID_LANGUAGES)))
	if src_lang not in VALID_LANGUAGES and src_lang != 'auto':
		raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (src_lang, ', '.join(VALID_LANGUAGES)))

	translator = get_translator(translator_key)

	if isinstance(translator, OfflineTranslator):
		if not translator.is_loaded():
			device = 'cuda' if use_cuda else 'cpu'
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
