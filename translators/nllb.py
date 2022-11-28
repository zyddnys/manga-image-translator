import re
from typing import List
from langdetect import detect
import huggingface_hub 

from translators.common import OfflineTranslator

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

class NLLBTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zho_Hans',
        'CHT': 'zho_Hant',
        'JPN': 'jpn_Jpan',
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
    _TRANSLATOR_MODEL = 'facebook/nllb-200-distilled-600M'

    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self._TRANSLATOR_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            detected_lang = detect('\n'.join(queries))
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang == None:
                print('Warning: Could not detect language from over all scentence. Will try per sentence.')
            else:
                from_lang = target_lang

        return [self._translate_sentence(from_lang, to_lang, query) for query in queries]

    def _translate_sentence(self, from_lang: str, to_lang: str, query: str) -> str:
        from transformers import pipeline

        if not self.is_loaded():
            return ''

        if from_lang == 'auto':
            detected_lang = detect(query)
            from_lang = self._map_detected_lang_to_translator(detected_lang)

        if from_lang == None:
            print(f'Warning: Offline Translation Failed. Could not detect language (Or language not supported for text: {query})')
            return ''

        translator = pipeline('translation', 
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=from_lang,
            tgt_lang=to_lang,
            max_length = 512,
        )

        result = translator(query)
        translated_text = self._clean_translation_output(result[0]['translation_text'])

        print(f'Offline Translation[{from_lang} -> {to_lang}] "{query}" -> "{translated_text}"')
        return translated_text

    def _clean_translation_output(self, text: str) -> str:
        '''
        Tries to spot and skim down invalid translations.
        '''
        words = text.split()
        elements = list(set(words))
        if len(elements) / len(words) < 0.1:
            words = words[:int(len(words) / 1.75)]
            text = ' '.join(words)

            # For words that appear more then four times consecutively, remove the excess
            for el in elements:
                el = re.escape(el)
                text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return text

    def _map_detected_lang_to_translator(self, lang):
        if not lang in ISO_639_1_TO_FLORES_200.keys():
            return None

        return ISO_639_1_TO_FLORES_200[lang]

    async def _download(self):
        # Preload models into cache as part of startup
        print(f'Detected offline translation mode. Pre-loading offline translation model: {self._TRANSLATOR_MODEL} ' +
              f'(This can take a long time as multiple GB\'s worth of data can be downloaded during this step)')
        huggingface_hub.snapshot_download(self._TRANSLATOR_MODEL)

    def _check_downloaded(self) -> bool:
        print(f'Detected cached model for offline translation: {self._TRANSLATOR_MODEL}')
        return huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'pytorch_model.bin') is not None

class NLLBBigTranslator(NLLBTranslator):
    _TRANSLATOR_MODEL = 'facebook/nllb-200-distilled-1.3B'
