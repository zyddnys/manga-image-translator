from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
import re

from translators.common import CommonTranslator

OFFLINE_TRANSLATOR_MODEL_MAP = {
    "offline": "facebook/nllb-200-distilled-600M",
    "offline_big": "facebook/nllb-200-distilled-1.3B",
}

LANGUAGE_CODE_MAP = {
	'CHS': 'zho_Hans',
	'CHT': 'zho_Hant',
	'JPN': "jpn_Jpan",
	'ENG': 'eng_Latn',
	'KOR': 'kor_Hang',
	'VIN': 'vie_Latn',
	'CSY': 'ces_Latn',
	'NLD': 'nld_Latn',
	'FRA': 'fra_Latn',
	'DEU': 'deu_Latn',
	'HUN': 'hun_Latn',
	'ITA': 'ita_Latn',
	'PLK': 'pol_Latn',
	'PTB': 'por_Latn',
	'ROM': 'ron_Latn',
	'RUS': 'rus_Cyrl',
	'ESP': 'spa_Latn',
	'TRK': 'tur_Latn',
}

ISO_639_1_TO_FLORES_200 = {
    'zh-cn': 'zho_Hans',
	'zh-tw': 'zho_Hant',
	'ja': 'jpn_Jpan',
	'en': 'eng_Latn',
	'kn': 'kor_Hang',
	'vi': 'vie_Latn',
	'cs': 'ces_Latn',
	'nl': 'nld_Latn',
	'fr': 'fra_Latn',
	'de': 'deu_Latn',
	'hu': 'hun_Latn',
	'it': 'ita_Latn',
	'pl': 'pol_Latn',
	'pt': 'por_Latn',
	'ro': 'ron_Latn',
	'ru': 'rus_Cyrl',
	'es': 'spa_Latn',
	'tr': 'tur_Latn',
}

class OfflineTranslator(CommonTranslator):
    def __init__(self):
        self.model_name = None
        self.loaded = False
        self.model = None
        self.tokenizer = None

    def load(self, translator, use_cuda):
        # Lazy load memory heavy models
        if not self.loaded:
            self.use_cuda = use_cuda
            self.model_name = OFFLINE_TRANSLATOR_MODEL_MAP[translator]
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.loaded = True

    def is_loaded(self):
        return self.loaded

    def _get_language_code(self, key):
        return LANGUAGE_CODE_MAP[key]

    async def _translate(self, from_lang, to_lang, queries):
        if from_lang == 'auto':
            detected_lang = detect('\n'.join(queries))
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang == None:
                print("Warning: Could not detect language from over all scentence. Will try per sentence.")
            else:
                from_lang = target_lang

        return [self.translate_sentence(from_lang, to_lang, query) for query in queries]

    def translate_sentence(self, from_lang, to_lang, query_text):
        if not self.is_loaded():
            return ""

        if from_lang == 'auto':
            detected_lang = detect(query_text)
            from_lang = self._map_detected_lang_to_translator(detected_lang)

        if from_lang == None:
            print(f"Warning: Offline Translation Failed. Could not detect language (Or language not supported for text: {query_text})")
            return ""

        translator = pipeline('translation', 
            device=0 if self.use_cuda else -1,
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=from_lang,
            tgt_lang=to_lang,
            max_length = 512,
        )

        result = translator(query_text)
        translated_text = self._clean_translation_output(result[0]['translation_text'])

        print(f"Offline Translation[{from_lang} -> {to_lang}] \"{query_text}\" -> \"{translated_text}\"")
        return translated_text

    def _clean_translation_output(self, text: str) -> str:
        words = text.split()
        elements = list(set(words))
        if len(elements) / len(words) < 0.1:
            words = words[:int(len(words) / 1.75)]
            text = ' '.join(words)

            # For parts that appear more then four times consecutively, remove the excess
            for el in elements:
                el = re.escape(el)
                text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return text

    def _map_detected_lang_to_translator(self, lang):
        if not lang in ISO_639_1_TO_FLORES_200.keys():
            return None

        return ISO_639_1_TO_FLORES_200[lang]
