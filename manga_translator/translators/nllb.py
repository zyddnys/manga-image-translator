from typing import List
import huggingface_hub
import langid

from .common import OfflineTranslator

ISO_639_1_TO_FLORES_200 = {
    'zh': 'zho_Hans',
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
    'uk': 'ukr_Cyrl',
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

        if ':' not in device:
            device += ':0'
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self._TRANSLATOR_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            detected_lang = langid.classify('\n'.join(queries))[0]
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang == None:
                self.logger.warn('Could not detect language from over all sentence. Will try per sentence.')
            else:
                from_lang = target_lang

        return [self._translate_sentence(from_lang, to_lang, query) for query in queries]

    def _translate_sentence(self, from_lang: str, to_lang: str, query: str) -> str:
        from transformers import pipeline

        if not self.is_loaded():
            return ''

        if from_lang == 'auto':
            detected_lang = langid.classify(query)[0]
            from_lang = self._map_detected_lang_to_translator(detected_lang)

        if from_lang == None:
            self.logger.warn(f'NLLB Translation Failed. Could not detect language (Or language not supported for text: {query})')
            return ''

        translator = pipeline('translation',
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=from_lang,
            tgt_lang=to_lang,
            max_length = 512,
        )

        result = translator(query)[0]['translation_text']
        return result

    def _map_detected_lang_to_translator(self, lang):
        if not lang in ISO_639_1_TO_FLORES_200:
            return None

        return ISO_639_1_TO_FLORES_200[lang]

    async def _download(self):
        huggingface_hub.snapshot_download(self._TRANSLATOR_MODEL)

    def _check_downloaded(self) -> bool:
        return huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'pytorch_model.bin') is not None

class NLLBBigTranslator(NLLBTranslator):
    _TRANSLATOR_MODEL = 'facebook/nllb-200-distilled-1.3B'
