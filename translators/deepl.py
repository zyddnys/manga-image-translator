
import deepl

from translators.common import CommonTranslator
from .keys import DEEPL_AUTH_KEY

LANGUAGE_CODE_MAP = {
	'CHS': 'ZH',
	'CHT': 'ZH',
	'JPN': "JA",
	'ENG': 'EN-US',
	'KOR': 'NONE',
	'VIN': 'NONE',
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
	'TRK': 'NONE',
}

class DeeplTranslator(CommonTranslator):
	def __init__(self):
		self.translator = deepl.Translator(DEEPL_AUTH_KEY)

	def _get_language_code(self, key):
		return LANGUAGE_CODE_MAP[key]

	async def _translate(self, from_lang, to_lang, queries):
		return self.translator.translate_text('\n'.join(queries), target_lang = to_lang).text.split('\n')


