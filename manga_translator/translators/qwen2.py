import os
import re
from typing import List, Dict
from omegaconf import OmegaConf

from ..config import TranslatorConfig
from .common import OfflineTranslator
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

# Adapted from:
# https://github.com/zyddnys/manga-image-translator/issues/680#issue-2428018275
# manga_translator/translators/chatgpt.py

class Qwen2Translator(OfflineTranslator, ConfigGPT):
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

    _TRANSLATOR_MODEL = "Qwen/Qwen2-1.5B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = False

    def __init__(self):
        OfflineTranslator.__init__(self)
        ConfigGPT.__init__(self, config_key='qwen2') 

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config

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
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self._TRANSLATOR_MODEL)

    async def _unload(self):
        del self.model
        del self.tokenizer

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        model_inputs = self.tokenize(queries, to_lang)
        # Generate the translation
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=10240
        )

        # Extract the generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        query_size = len(queries)

        translations = []
        self.logger.debug('-- Qwen2 Response --\n' + response)
        new_translations = re.split(r'<\|\d+\|>', response)

        # When there is only one query chatgpt likes to exclude the <|1|>
        if not new_translations[0].strip():
            new_translations = new_translations[1:]

        if len(new_translations) <= 1 and query_size > 1:
            # Try splitting by newlines instead
            new_translations = re.split(r'\n', response)

        if len(new_translations) > query_size:
            new_translations = new_translations[: query_size]
        elif len(new_translations) < query_size:
            new_translations = new_translations + [''] * (query_size - len(new_translations))

        translations.extend([t.strip() for t in new_translations])

        return translations

    def tokenize(self, queries, to_lang):
        prompt = f"""Translate into {to_lang} and keep the original format.\n"""
        prompt += '\nOriginal:'
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1}|>{query}'

        tokenizer = self.tokenizer
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]
        
        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})
            
        messages.append({'role': 'user', 'content': prompt})

        self.logger.debug("-- Qwen2 prompt --\n" + 
                "\n".join(f"{msg['role'].capitalize()}:\n {msg['content']}" for msg in messages) +
                "\n"
            )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Ensure pad_token is set correctly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True
        ).to(self.device)

        return model_inputs


class Qwen2BigTranslator(Qwen2Translator):
    _TRANSLATOR_MODEL = "Qwen/Qwen2-7B-Instruct"
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_DIR, OfflineTranslator._MODEL_SUB_DIR, _TRANSLATOR_MODEL)
    _IS_4_BIT = True
