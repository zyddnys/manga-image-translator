from book_maker.translator.gpt3_translator import GPT3
# from book_maker.translator.chatgptapi_translator import ChatGPTAPI

from .common import CommonTranslator, MissingAPIKeyException
from .keys import GPT3_AUTH_KEY

class GPT3Translator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
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
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
    }

    def __init__(self):
        super().__init__()
        if not GPT3_AUTH_KEY:
            raise MissingAPIKeyException('Please set the GPT3_AUTH_KEY environment variable before using the gpt3 translator.')

    async def _translate(self, from_lang, to_lang, queries):
        self.translator = GPT3(GPT3_AUTH_KEY, to_lang)
        return self.translator.translate('\n'.join(queries), target_lang = to_lang).text.split('\n')
