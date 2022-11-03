from translators.common import CommonTranslator
class NoTranslator(CommonTranslator):
	_LANGUAGE_CODE_MAP = {
		'CHS': 0,
		'CHT': 0,
		'JPN': 0,
		'KOR': 0,
		'ENG': 0,
		'CSY': 0,
		'NLD': 0,
		'FRA': 0,
		'DEU': 0,
		'HUN': 0,
		'ITA': 0,
		'PLK': 0,
		'PTB': 0,
		'ROM': 0,
		'RUS': 0,
		'ESP': 0,
		'TRK': 0,
		'VIN': 0,
	}
	async def _translate(self, from_lang, to_lang, queries):
		return ''