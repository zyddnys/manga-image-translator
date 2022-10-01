from cgitb import reset
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

OFFLINE_TRANSLATOR_MODEL_MAP = {
    "offline": "facebook/nllb-200-distilled-600M",
    "offline_big": "facebook/nllb-200-distilled-1.3B",
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

class Translator(object):
    def __init__(self):
        self.model_name = None
        self.loaded = False
        self.model = None
        self.tokenizer = None

    def load(self, translator):
        # Lazy load memory heavy models
        if not self.loaded:
            self.model_name = OFFLINE_TRANSLATOR_MODEL_MAP[translator]
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.loaded = True

    def is_loaded(self):
        return self.loaded

    async def translate(self, from_lang, to_lang, query_text):
        if from_lang == 'auto':
            detected_lang = detect(query_text)
            target_lang = self._map_detected_lang_to_translator(detected_lang)

            if target_lang == None:
                print("Warning: Could not detect language from over all scentence. Will try per scentece.")
            else:
                from_lang = target_lang

        return [self.translate_sentence(from_lang, to_lang, text) for text in query_text.split('\n')]

    def translate_sentence(self, from_lang, to_lang, query_text) :
        if not self.is_loaded():
            return ""

        if from_lang == 'auto':
            detected_lang = detect(query_text)
            from_lang = self._map_detected_lang_to_translator(detected_lang)

        if from_lang == None:
            print(f"Warning: Offline Translation Failed. Could not detect language (Or language not supported for text: {query_text})")
            return ""

        translator = pipeline('translation', 
            device=self._get_device(),
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=from_lang,
            tgt_lang=to_lang,
            max_length = 512
        )

        result = translator(query_text)
        translated_text = result[0]['translation_text']
        print(f"Offline Translation[{from_lang} -> {to_lang}] \"{query_text}\" -> \"{translated_text}\"")
        return translated_text

    def _map_detected_lang_to_translator(self, lang):
        if not lang in ISO_639_1_TO_FLORES_200.keys():
            return None

        return ISO_639_1_TO_FLORES_200[lang]
    
    @staticmethod
    def _get_device():
        # -1: CPU
        # 0: CUDA DEVICE 0
        return 0 if torch.cuda.is_available() and os.getenv('USE_CUDA_FOR_OFFLINE_TRANSLATION', False) else -1
