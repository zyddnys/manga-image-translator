from typing import List, Tuple
from abc import abstractmethod
from functools import cached_property
import os

from utils import ModelWrapper

class CommonTranslator():
    _LANGUAGE_CODE_MAP = {}

    @cached_property
    def supported_src_languages(self) -> List[str]:
        return ['auto'] + list(self._LANGUAGE_CODE_MAP)

    @cached_property
    def supported_tgt_languages(self) -> List[str]:
        return list(self._LANGUAGE_CODE_MAP)

    def _parse_language_code(self, key: str) -> str:
        return self._LANGUAGE_CODE_MAP[key]

    def parse_language_codes(self, from_lang: str, to_lang: str) -> Tuple[str, str]:
        try:
            _from_lang = self._parse_language_code(from_lang) if from_lang != 'auto' else 'auto'
            if not _from_lang:
                raise KeyError(from_lang)
            _to_lang = self._parse_language_code(to_lang)
            if not _to_lang:
                raise KeyError(to_lang)
            return _from_lang, _to_lang
        except KeyError as e:
            print('Language not supported for chosen translator: "%s". Supported languages: "%s"' % (e, ','.join(self._LANGUAGE_CODE_MAP)))
            raise e

    async def translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        '''
        Translates list of queries of one language into another.
        '''
        if from_lang == to_lang:
            return []
        return await self._translate(*self.parse_language_codes(from_lang, to_lang), queries)

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
