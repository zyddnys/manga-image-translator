from typing import Callable, List
import langid

from .common import OfflineTranslator, ISO_639_1_TO_VALID_LANGUAGES
from .m2m100 import M2M100Translator
from .sugoi import SugoiTranslator


get_translator: Callable[[str], OfflineTranslator] = None

def prepare(translator_supplicant: Callable[[str], OfflineTranslator]):
    global get_translator
    get_translator = translator_supplicant

class SelectiveOfflineTranslator(OfflineTranslator):
    '''
    Translator that automatically chooses most suitable offline variant for
    specific language.
    `load` and `download` calls are cached until `forward` is called.
    '''

    _LANGUAGE_CODE_MAP = {
        **M2M100Translator._LANGUAGE_CODE_MAP,
        **SugoiTranslator._LANGUAGE_CODE_MAP,
    }

    def __init__(self):
        super().__init__()
        self._cached_load_params = None
        self._real_translator: OfflineTranslator = None

    def select_translator(self, from_lang: str, to_lang: str) -> OfflineTranslator:
        if from_lang != 'auto':
            sugoi_translator = get_translator('sugoi')
            if sugoi_translator.supports_languages(from_lang, to_lang):
                return sugoi_translator
        return get_translator('m2m100_big')

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool) -> List[str]:
        if from_lang == 'auto':
            detected_lang = langid.classify(' '.join(queries))[0]
            if detected_lang in ISO_639_1_TO_VALID_LANGUAGES:
                from_lang = ISO_639_1_TO_VALID_LANGUAGES[detected_lang]

        self._real_translator = self.select_translator(from_lang, to_lang)
        self.logger.info(f'Selected translator: {self._real_translator.__class__.__name__}')

        if self._cached_load_params:
            await self._real_translator.load(*self._cached_load_params)
            self._cached_load_params = None

        return await self._real_translator.translate(from_lang, to_lang, queries, use_mtpe)

    async def load(self, from_lang: str, to_lang: str, device: str):
        self._cached_load_params = [from_lang, to_lang, device]

    async def reload(self, from_lang: str, to_lang: str, device: str):
        self._cached_load_params = [from_lang, to_lang, device]

    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def _unload(self):
        pass

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

# class SelectiveBigOfflineTranslator(SelectiveOfflineTranslator):
#     def select_translator(self, from_lang: str, to_lang: str) -> OfflineTranslator:
#         if from_lang != 'auto':
#             sugoi_translator = get_translator('sugoi')
#             if sugoi_translator.supports_languages(from_lang, to_lang):
#                 return sugoi_translator
#         return get_translator('m2m100_big')
