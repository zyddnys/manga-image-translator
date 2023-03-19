import re
import openai

from .common import CommonTranslator, MissingAPIKeyException

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

        self.prompt_template = 'Please help me to translate the following queries to {language}:{text}'

    async def _translate(self, from_lang, to_lang, queries):
        text = ''
        for i, query in enumerate(queries):
            text += f'\n\nQuery {i+1}:\n---------------\n{query}'
        prompt = self.prompt_template.format(text=text, language=to_lang)
        print(prompt)

        completion = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            max_tokens=1024,
            temperature=1,
        )
        text = completion.choices[0].text
        print(text)
        print()
        translations = re.findall(r'\n*(?:Query|Answer) \d+:(?:\n-+\n)?(.*)', text)
        translations = [t.strip() for t in translations]
        print(translations)
        return translations
