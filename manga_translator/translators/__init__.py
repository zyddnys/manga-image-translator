import langid

from .common import *
from .baidu import BaiduTranslator
from .google import GoogleTranslator
from .youdao import YoudaoTranslator
from .deepl import DeeplTranslator
from .papago import PapagoTranslator
from .chatgpt import GPT3Translator, GPT35TurboTranslator
from .nllb import NLLBTranslator, NLLBBigTranslator
from .sugoi import JparacrawlTranslator, JparacrawlBigTranslator, SugoiTranslator
from .m2m100 import M2M100Translator, M2M100BigTranslator
from .selective import SelectiveOfflineTranslator, prepare as prepare_selective_translator
from .none import NoneTranslator
from .original import OriginalTranslator

OFFLINE_TRANSLATORS = {
    'offline': SelectiveOfflineTranslator,
    'nllb': NLLBTranslator,
    'nllb_big': NLLBBigTranslator,
    'sugoi': SugoiTranslator,
    'jparacrawl': JparacrawlTranslator,
    'jparacrawl_big': JparacrawlBigTranslator,
    'm2m100': M2M100Translator,
    'm2m100_big': M2M100BigTranslator,
}

TRANSLATORS = {
    'google': GoogleTranslator,
    'youdao': YoudaoTranslator,
    'baidu': BaiduTranslator,
    'deepl': DeeplTranslator,
    'papago': PapagoTranslator,
    'gpt3': GPT3Translator,
    'gpt3.5': GPT35TurboTranslator,
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

class TranslatorChain():
    def __init__(self, string: str):
        """
        Parses string in form 'trans1:lang1;trans2:lang2' into chains,
        which will be executed one after another when passed to the dispatch function.
        """
        if not string:
            raise Exception('Invalid translator chain')
        self.chain = []
        self.target_lang = None
        for g in string.split(';'):
            trans, lang = g.split(':')
            if trans not in TRANSLATORS:
                raise ValueError(f'Invalid choice: %s (choose from %s)' % (trans, ', '.join(map(repr, TRANSLATORS))))
            if lang not in VALID_LANGUAGES:
                raise ValueError(f'Invalid choice: %s (choose from %s)' % (lang, ', '.join(map(repr, VALID_LANGUAGES))))
            self.chain.append((trans, lang))
        self.translators, self.langs = list(zip(*self.chain))

async def prepare(chain: TranslatorChain):
    for key, tgt_lang in chain.chain:
        translator = get_translator(key)
        translator.supports_languages('auto', tgt_lang, fatal=True)
        if isinstance(translator, OfflineTranslator):
            await translator.download()

# TODO: Optionally take in strings instead of TranslatorChain for simplicity
async def dispatch(chain: TranslatorChain, queries: List[str], use_mtpe: bool = False, device: str = 'cpu') -> List[str]:
    if not queries:
        return queries
    if chain.target_lang is not None:
        text_lang = ISO_639_1_TO_VALID_LANGUAGES.get(langid.classify('\n'.join(queries))[0])
        translator = None
        for key, lang in chain.chain:
            if text_lang == lang:
                translator = get_translator(key)
                break
        if translator is None:
            translator = get_translator(chain.langs[0])
        if isinstance(translator, OfflineTranslator):
            await translator.load('auto', chain.target_lang, device)
        queries = await translator.translate('auto', chain.target_lang, queries, use_mtpe)
        return queries

    for key, tgt_lang in chain.chain:
        translator = get_translator(key)
        if isinstance(translator, OfflineTranslator):
            await translator.load('auto', tgt_lang, device)
        queries = await translator.translate('auto', tgt_lang, queries, use_mtpe)
    return queries
