from functools import cached_property
from typing import List, Tuple
from abc import abstractmethod
import os

from utils import ModelWrapper

class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None, *args: object) -> None:
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error, *args)

class CommonTranslator():
    _LANGUAGE_CODE_MAP = {}

    @cached_property
    def supported_src_languages(self) -> List[str]:
        return ['auto'] + list(self._LANGUAGE_CODE_MAP)

    @cached_property
    def supported_tgt_languages(self) -> List[str]:
        return list(self._LANGUAGE_CODE_MAP)

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False):
        if from_lang not in self.supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, self.supported_src_languages)
            return False
        if to_lang not in self.supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, self.supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        '''
        Translates list of queries of one language into another.
        '''
        if from_lang == to_lang:
            return []
        return await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries)

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'translators')

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        return await self.forward(from_lang, to_lang, queries)

    @abstractmethod
    async def _forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)
