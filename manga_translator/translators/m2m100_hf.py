import os
from typing import List
from langdetect import detect

from .common import OfflineTranslator

ISO_639_1_TO_M2M100 = {
    'zh': 'zh', 'cs': 'cs', 'nl': 'nl', 'en': 'en', 'fr': 'fr', 'de': 'de',
    'hu': 'hu', 'it': 'it', 'ja': 'ja', 'ko': 'ko', 'pl': 'pl', 'pt': 'pt',
    'ro': 'ro', 'ru': 'ru', 'es': 'es', 'tr': 'tr', 'uk': 'uk', 'vi': 'vi',
    'ar': 'ar', 'sr': 'sr', 'hr': 'hr', 'th': 'th', 'id': 'id'
}

class M2M100HFTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh', 'CHT': 'zh', 'CSY': 'cs', 'NLD': 'nl', 'ENG': 'en',
        'FRA': 'fr', 'DEU': 'de', 'HUN': 'hu', 'ITA': 'it', 'JPN': 'ja',
        'KOR': 'ko', 'PLK': 'pl', 'PTB': 'pt', 'ROM': 'ro', 'RUS': 'ru',
        'ESP': 'es', 'TRK': 'tr', 'UKR': 'uk', 'VIN': 'vi', 'ARA': 'ar',
        'SRP': 'sr', 'HRV': 'hr', 'THA': 'th', 'IND': 'id'
    }
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'm2m100')
    _TRANSLATOR_MODEL = 'facebook/m2m100_418M'

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
            try:
                detected = detect('\n'.join(queries))
                from_lang = ISO_639_1_TO_M2M100.get(detected)
            except Exception as e:
                self.logger.warning(f'Language detection failed: {e}')
                from_lang = None

            if from_lang is None:
                self.logger.warning('Could not detect source language, skipping translation')
                return [''] * len(queries)

        return self._translate_batch(from_lang, to_lang, queries)

    def _translate_batch(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        import torch
        try:
            self.tokenizer.src_lang = from_lang
            encoded = self.tokenizer(queries, return_tensors='pt', padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(to_lang),
                    max_length=512,
                    num_beams=5,
                )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f'Batch translation failed: {e}')
            return [''] * len(queries)

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

class M2M100HFBigTranslator(M2M100HFTranslator):
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, 'm2m100')
    _TRANSLATOR_MODEL = 'facebook/m2m100_1.2B'
