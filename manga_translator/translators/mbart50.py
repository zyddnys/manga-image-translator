import os
import py3langid as langid


from .common import OfflineTranslator

ISO_639_1_TO_MBart50 = {

    'ar': 'ar_AR',
    'de': 'de_DE',
    'en': 'en_XX',
    'es': 'es_XX',
    'fr': 'fr_XX',
    'hi': 'hi_IN',
    'it': 'it_IT',
    'ja': 'ja_XX',
    'nl': 'nl_XX',
    'pl': 'pl_PL',
    'pt': 'pt_XX',
    'ru': 'ru_RU',
    'sw': 'sw_KE',
    'th': 'th_TH',
    'tr': 'tr_TR',
    'ur': 'ur_PK',
    'vi': 'vi_VN',
    'zh': 'zh_CN',
    
    
}

class MBart50Translator(OfflineTranslator):
    # https://huggingface.co/facebook/mbart-large-50
    # other languages can be added as well
    _LANGUAGE_CODE_MAP = {
        "ARA": "ar_AR",
        "DEU": "de_DE",
        "ENG": "en_XX",
        "ESP": "es_XX",
        "FRA": "fr_XX",
        "HIN": "hi_IN",
        "ITA": "it_IT",
        "JPN": "ja_XX",
        "NLD": "nl_XX",
        "PLK": "pl_PL",
        "PTB": "pt_XX",
        "RUS": "ru_RU",
        "SWA": "sw_KE",
        "THA": "th_TH",
        "TRK": "tr_TR",
        "URD": "ur_PK",
        "VIN": "vi_VN",
        "CHS": "zh_CN",
    }
    
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'mbart50')
    
    _TRANSLATOR_MODEL = "facebook/mbart-large-50-many-to-many-mmt"



    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import (
            MBartForConditionalGeneration,
            AutoTokenizer,
        )
        if ':' not in device:
            device += ':0'
        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(self._TRANSLATOR_MODEL)
        if self.device != 'cpu':
            self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: list[str]) -> list[str]:
        if from_lang == 'auto':
            detected_lang = langid.classify('\n'.join(queries))[0]
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang == None:
                self.logger.warn('Could not detect language from over all sentence. Will try per sentence.')
            else:
                from_lang = target_lang

        return [self._translate_sentence(from_lang, to_lang, query) for query in queries]

    def _translate_sentence(self, from_lang: str, to_lang: str, query: str) -> str:

        if not self.is_loaded():
            return ''

        if from_lang == 'auto':
            detected_lang = langid.classify(query)[0]
            from_lang = self._map_detected_lang_to_translator(detected_lang)

        if from_lang == None:
            self.logger.warn(f'MBart50 Translation Failed. Could not detect language (Or language not supported for text: {query})')
            return ''
        
        self.tokenizer.src_lang = from_lang 
        tokens = self.tokenizer(query, return_tensors="pt")
        # move to device
        if self.device != 'cpu':
            tokens = tokens.to(self.device)
        generated_tokens = self.model.generate(**tokens, forced_bos_token_id=self.tokenizer.lang_code_to_id[to_lang])
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result

    def _map_detected_lang_to_translator(self, lang):
        if lang not in ISO_639_1_TO_MBart50:
            return None

        return ISO_639_1_TO_MBart50[lang]

    async def _download(self):
        import huggingface_hub
        # do not download msgpack and h5 files as they are not needed to run the model
        huggingface_hub.snapshot_download(self._TRANSLATOR_MODEL, cache_dir=self._MODEL_SUB_DIR, ignore_patterns=["*.msgpack", "*.h5", '*.ot',".*", "*.safetensors"])

    def _check_downloaded(self) -> bool:
        import huggingface_hub
        return huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'pytorch_model.bin', cache_dir=self._MODEL_SUB_DIR) is not None