import re
from typing import List, Tuple
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper, repeating_sequence

try:
    import readline
except Exception:
    readline = None

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

ISO_639_1_TO_VALID_LANGUAGES = {
        'zh': 'CHS',
        'ja': 'JPN',
        'en': 'ENG',
        'ko': 'KOR',
        'vi': 'VIN',
        'cs': 'CSY',
        'nl': 'NLD',
        'fr': 'FRA',
        'de': 'DEU',
        'hu': 'HUN',
        'it': 'ITA',
        'pl': 'PLK',
        'pt': 'PTB',
        'ro': 'ROM',
        'ru': 'RUS',
        'es': 'ESP',
        'tr': 'TRK',
        'uk': 'UKR',
}

class InvalidServerResponse(Exception):
    pass

class MissingAPIKeyException(Exception):
    pass

class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error)

class MTPEAdapter():
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        # TODO: Make it work in windows (e.g. through os.startfile)
        if not readline:
            print('MTPE is currently only supported on linux')
            return translations
        new_translations = []
        print('Running Machine Translation Post Editing (MTPE)')
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f'\n[{i + 1}/{len(queries)}] {query}:')
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace('\n', '\\n')))
            new_translation = ''
            try:
                new_translation = input(' -> ').replace('\\n', '\n')
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        print()
        return new_translations

class CommonTranslator(InfererModule):
    _LANGUAGE_CODE_MAP = {}
    _INVALID_REPEAT_COUNT = 0

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ['auto'] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False) -> List[str]:
        """
        Translates list of queries of one language into another.
        """
        if to_lang not in VALID_LANGUAGES:
            raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (to_lang, ', '.join(VALID_LANGUAGES)))
        if from_lang not in VALID_LANGUAGES and from_lang != 'auto':
            raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (from_lang, ', '.join(VALID_LANGUAGES)))
        self.logger.info(f'Translating into {VALID_LANGUAGES[to_lang]}')

        translations = [''] * len(queries)

        if from_lang == to_lang:
            return translations

        # TODO: simplify handling of _INVALID_REPEAT_COUNT because too messy

        unchecked_indices = list(range(len(queries))) # unchecked for invalid translations
        for i in range(1 + self._INVALID_REPEAT_COUNT):
            _translations = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries)

            # Only overwrite invalid translations
            if len(unchecked_indices) > 0:
                for j in unchecked_indices:
                    if j < len(_translations):
                        translations[j] = _translations[j]
            else:
                translations = _translations

            # Extend returned translations to have the same size as queries
            if len(translations) < len(queries):
                translations.extend([''] * (len(queries) - len(translations)))
            elif len(translations) > len(queries):
                translations = translations[:len(queries)]

            if self._INVALID_REPEAT_COUNT <= 0:
                break

            # Repeat invalid translations with slightly modified queries
            n_unchecked_indices = []
            for j in unchecked_indices:
                q, t = queries[j], translations[j]
                if self._is_translation_invalid(q, t):
                    n_unchecked_indices.append(j)
                    queries[j] = self._modify_invalid_translation_query(q, t)
            if not n_unchecked_indices:
                break
            unchecked_indices = n_unchecked_indices
            self.logger.info('Repeating because of invalid translation')

        translations = [self._clean_translation_output(q, r) for q, r in zip(queries, translations)]
        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)
        return translations

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        if not query or not trans:
            return False

        query_symbols_count = len(set(query))
        trans_symbols_count = len(set(trans))
        if query_symbols_count > 6 and trans_symbols_count < 0.3 * query_symbols_count:
            return True
        return False

    def _modify_invalid_translation_query(self, query: str, trans: str) -> str:
        """
        Should be overwritten if _INVALID_REPEAT_COUNT was set. It modifies the query
        for the next translation attempt.
        """
        raise NotImplementedError()

    def _clean_translation_output(self, query: str, trans: str) -> str:
        """
        Tries to spot and skim down invalid translations.
        """
        if not query or not trans:
            return ''

        # '  ' -> ' '
        trans = re.sub(r'\s+', r' ', trans)
        # 'text .' -> 'text.'
        trans = re.sub(r'\s+([.,;])', r'\1', trans)

        seq = repeating_sequence(trans.lower())

        # 'aaaaaaaaaaaaa' -> 'aaaaaa'
        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            # Shrink sequence to length of original query
            trans = seq * max(len(query) // len(seq), 1)
            # Transfer capitalization of query to translation
            nTrans = ''
            for i in range(min(len(trans), len(query))):
                nTrans += trans[i].upper() if query[i].isupper() else trans[i]
            trans = nTrans

        # ' ! ! . . ' -> ' !!.. '
        trans = re.sub(r'([.!?])\s+(?=[.!?]|$)', r'\1', trans)

        # words = text.split()
        # elements = list(set(words))
        # if len(elements) / len(words) < 0.1:
        #     words = words[:int(len(words) / 1.75)]
        #     text = ' '.join(words)

        #     # For words that appear more then four times consecutively, remove the excess
        #     for el in elements:
        #         el = re.escape(el)
        #         text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return trans

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_SUB_DIR = 'translators'

    async def _translate(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)
