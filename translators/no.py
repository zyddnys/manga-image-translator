from translators.common import CommonTranslator

class NoTranslator(CommonTranslator):
	_LANGUAGE_CODE_MAP = {
		'CHS': 'no',
		'CHT': 'no',
		'JPN': 'no',
		'KOR': 'no',
		'ENG': 'no',
		'CSY': 'no',
		'NLD': 'No',
		'FRA': 'no',
		'DEU': 'no',
		'HUN': 'no',
		'ITA': 'no',
		'PLK': 'no',
		'PTB': 'no',
		'ROM': 'no',
		'RUS': 'no',
		'ESP': 'no',
		'TRK': 'no',
		'VIN': 'no',
	}
	async def _translate(self, from_lang, to_lang, queries):
		return " "


