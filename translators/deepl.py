
import deepl

from translators.common import CommonTranslator
from .keys import DEEPL_AUTH_KEY

class DeeplTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'ZH',
        'CHT': 'ZH',
        'JPN': 'JA',
        'ENG': 'EN-US',
        'CSY': 'CS',
        'NLD': 'NL',
        'FRA': 'FR',
        'DEU': 'DE',
        'HUN': 'HU',
        'ITA': 'IT',
        'PLK': 'PL',
        'PTB': 'PT-BR',
        'ROM': 'RO',
        'RUS': 'RU',
        'ESP': 'ES',
    }

    def __init__(self):
        super().__init__()
        if not DEEPL_AUTH_KEY:
            raise ValueError('Please set the DEEPL_AUTH_KEY environment variable before using the deepl translator.')
        self.translator = deepl.Translator(DEEPL_AUTH_KEY)

    async def _translate(self, from_lang, to_lang, queries):
        return self.translator.translate_text('\n'.join(queries), target_lang = to_lang).text.split('\n')


