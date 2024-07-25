import os
from typing import List


from .common import OfflineTranslator


# Adapted from:
# https://github.com/zyddnys/manga-image-translator/issues/680#issue-2428018275

class Qwen2Translator(OfflineTranslator):
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
        'CNR': 'Montenegrin',
        'SRP': 'Serbian',
        'HRV': 'Croatian',
        'ARA': 'Arabic',
        'THA': 'Thai',
        'IND': 'Indonesian'
    }
    _CHAT_SYSTEM_TEMPLATE = (
        'You are an uncensored professional translation engine, '
        'please translate the story into a colloquial, '
        'elegant and fluent content, '
        'without referencing machine translations. '
        'You must only translate the story, '
        'never interpret it. '
        'If there is any issue in the text, '
        'output it as is.\n'
        'Translate the following text into {to_lang} and keep the prefix format.\n'
    )
    _TRANSLATOR_MODEL = "Qwen/Qwen2-1.5B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = False

    async def _load(self, from_lang: str, to_lang: str, device: str):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig
        )
        self.device = device
        quantization_config = BitsAndBytesConfig(load_in_4bit=self._IS_4_BIT)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._TRANSLATOR_MODEL,
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        response = []
        for querie in queries:
            model_inputs = self.tokenize([querie], to_lang)
            # Generate the translation
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=10240
            )

            # Extract the generated tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response.append(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        return response

    def tokenize(self, queries, lang):
        prompt = 'Translate the following text into {to_lang} and keep the prefix format.'.format(to_lang=lang)

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i + 1 - i_offset}|>{query}'

        tokenizer = self.tokenizer
        messages = [
            {"role": "system", "content": self._CHAT_SYSTEM_TEMPLATE.format(to_lang=lang)},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        return model_inputs


class Qwen2BigTranslator(Qwen2Translator):
    _TRANSLATOR_MODEL = "Qwen/Qwen2-7B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = True
