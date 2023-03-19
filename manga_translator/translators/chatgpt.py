import re
import openai

from .common import CommonTranslator, MissingAPIKeyException

# Example query:
"""Please help me to translate the following queries to english:
Query 1:
ちょっと悪いんだけど

Query 2:
そこの魔法陣に入って頂戴

Query 3:
いやいや何も起きないから(嘘)
是否离线可用

Query 4:
いやいや何も起きないから(嘘)
是否离线可用

Translation 1:

"""

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
        if not openai.api_key:
            raise MissingAPIKeyException('Please set the OPENAI_API_KEY environment variable before using the chatgpt translator.')

    async def _translate(self, from_lang, to_lang, queries):
        prompt = f'Please help me to translate the following queries from a manga to {to_lang}:\n'
        for i, query in enumerate(queries):
            prompt += f'\nQuery {i+1}:\n{query}\n'
        prompt += '\nTranslation 1:\n'
        print(prompt)

        completion = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=1024,
            temperature=1,
        )
        response = completion.choices[0].text
        print(response)
        translations = re.split(r'Translation \d+:\n', response)
        translations = [t.strip() for t in translations]
        print(translations)
        return translations
