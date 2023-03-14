import os
import ctranslate2
import sentencepiece as spm
from typing import List

from .common import OfflineTranslator

# Adapted from:
# https://gist.github.com/ymoslem/a414a0ead0d3e50f4d7ff7110b1d1c0d
# https://github.com/ymoslem/DesktopTranslator

class M2M100Translator(OfflineTranslator):
    # Refer to https://github.com/ymoslem/DesktopTranslator/blob/main/utils/m2m_languages.json
    # other languages can be added as well
    _LANGUAGE_CODE_MAP = {
        'CHS': '__zh__',
        'CHT': '__zh__',
        'CSY': '__cs__',
        'NLD': '__nl__',
        'ENG': '__en__',
        'FRA': '__fr__',
        'DEU': '__de__',
        'HUN': '__hu__',
        'ITA': '__it__',
        'JPN': '__ja__',
        'KOR': '__ko__',
        'PLK': '__pl__',
        'PTB': '__pt__',
        'ROM': '__ro__',
        'RUS': '__ru__',
        'ESP': '__es__',
        'TRK': '__tr__',
        'UKR': '__uk__',
        'VIN': '__vi__',
    }
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_SUB_DIR, 'm2m_100')
    _CT2_MODEL_DIR = 'm2m100_418m'
    _MODEL_MAPPING = {
        'models': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/m2m100_418m_ct2.zip',
            'hash': '8a9cd0e00505a7879f26e5a1b396b447bc29967783a1e17e8df5eecb0c13d1c3',
            'archive': {
                'm2m100_418m/': '.',
            },
        },
    }

    async def _load(self, from_lang: str, to_lang: str, device: str):
        self.load_params = {
            'from_lang': from_lang,
            'to_lang': to_lang,
            'device': device,
        }
        self.model = ctranslate2.Translator(
            model_path=self._get_file_path(self._CT2_MODEL_DIR),
            device=device,
            device_index=0,
        )
        self.model.load_model()
        self.sentence_piece_processor = spm.SentencePieceProcessor(model_file=self._get_file_path(self._CT2_MODEL_DIR, 'sentencepiece.model'))

    async def _unload(self):
        self.model.unload_model()
        del self.model
        del self.sentence_piece_processor

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        queries_tokenized = self.tokenize(queries, from_lang)
        translated_tokenized = self.model.translate_batch(
            source=queries_tokenized,
            target_prefix=[[to_lang]] * len(queries),
            beam_size=5,
            max_batch_size=1024,
            return_alternatives=False,
            disable_unk=True,
            replace_unknowns=True,
            repetition_penalty=1.2,
        )
        translated = self.detokenize(list(map(lambda t: t[0]['tokens'], translated_tokenized)), to_lang)
        return translated

    def tokenize(self, queries, lang):
        sp = self.sentence_piece_processor
        if isinstance(queries, list):
            return sp.encode(queries, out_type=str)
        else:
            return [sp.encode(queries, out_type=str)]

    def detokenize(self, queries, lang):
        sp = self.sentence_piece_processor
        translation = sp.decode(queries)
        prefix_len = len(lang) + 1
        translation = [''.join(query)[prefix_len:] for query in translation]
        return translation

class M2M100BigTranslator(M2M100Translator):
    _MODEL_SUB_DIR = os.path.join(OfflineTranslator._MODEL_SUB_DIR, 'm2m_100')
    _CT2_MODEL_DIR = 'm2m100_12b'
    _MODEL_MAPPING = {
        'models': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/m2m100_12b_ct2.zip',
            'hash': '742d5380c2837affd3680339145d37fc78f537ad633958347b76e9be9c577662',
            'archive': {
                'm2m100_12b/': '.',
            },
        },
    }
