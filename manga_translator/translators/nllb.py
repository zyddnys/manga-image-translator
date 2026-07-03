import os
from typing import List
import py3langid as langid

from .common import OfflineTranslator

# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
ISO_639_1_TO_FLORES_200 = {
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'en': 'eng_Latn',
    'kn': 'kor_Hang',
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
    'uk': 'ukr_Cyrl',
    'vi': 'vie_Latn',
    'ar': 'arb_Arab',
    'sr': 'srp_Cyrl',
    'hr': 'hrv_Latn',
    'th': 'tha_Thai',
    'id': 'ind_Latn'
}

class NLLBTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zho_Hans',
        'CHT': 'zho_Hant',
        'JPN': 'jpn_Jpan',
        'ENG': 'eng_Latn',
        'KOR': 'kor_Hang',
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
        'UKR': 'ukr_Cyrl',
        'VIN': 'vie_Latn',
        'ARA': 'arb_Arab',
        'SRP': 'srp_Cyrl',
        'HRV': 'hrv_Latn',
        'THA': 'tha_Thai',
        'IND': 'ind_Latn'
    }
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'nllb')
    _TRANSLATOR_MODEL = 'facebook/nllb-200-distilled-600M'

    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        if device not in ('cpu',) and ':' not in device:
            device = device + ':0'
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self._TRANSLATOR_MODEL)
        self.model = self.model.to(device)
        self.model.eval()

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            detected_lang = langid.classify('\n'.join(queries))[0]
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang is None:
                self.logger.warning('Could not detect source language, skipping translation')
                return [''] * len(queries)
            from_lang = target_lang

        return self._translate_batch(from_lang, to_lang, queries)

    def _translate_batch(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        import torch
        try:
            self.tokenizer.src_lang = from_lang
            encoded = self.tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            target_lang_id = self.tokenizer.lang_code_to_id[to_lang]
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=target_lang_id,
                    max_length=512,
                    num_beams=5,
                )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f'Batch translation failed: {e}')
            return [''] * len(queries)

    def _map_detected_lang_to_translator(self, lang):
        if lang not in ISO_639_1_TO_FLORES_200:
            return None
        return ISO_639_1_TO_FLORES_200[lang]

    async def _download(self):
        import huggingface_hub
        huggingface_hub.snapshot_download(
            self._TRANSLATOR_MODEL,
            cache_dir=self._MODEL_SUB_DIR,
            ignore_patterns=['*.msgpack', '*.h5', '*.ot', '.*'],
        )

    def _check_downloaded(self) -> bool:
        import huggingface_hub
        return (
            huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'model.safetensors', cache_dir=self._MODEL_SUB_DIR) is not None
            or huggingface_hub.try_to_load_from_cache(self._TRANSLATOR_MODEL, 'pytorch_model.bin', cache_dir=self._MODEL_SUB_DIR) is not None
        )

class NLLBBigTranslator(NLLBTranslator):
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'nllb_big')
    _TRANSLATOR_MODEL = 'facebook/nllb-200-distilled-1.3B'
