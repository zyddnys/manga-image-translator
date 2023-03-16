import ctranslate2
import sentencepiece as spm
from typing import List
import re

from .common import OfflineTranslator
from ..utils import chunks

class JparacrawlTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'JPN': 'ja',
        'ENG': 'en',
    }
    _CT2_MODEL_FOLDERS = {
        'ja-en': 'jparacrawl/base-ja-en',
        'en-ja': 'jparacrawl/base-en-ja',
    }
    _MODEL_MAPPING = {
        'models': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/jparacrawl-base-models.zip',
            'hash': 'e98e0fa35a80d2bc48c16673914639db66da1013ec66cc7b79119cdd3b542ebb',
            'archive': {
                'spm.ja.nopretok.model': 'jparacrawl/',
                'spm.en.nopretok.model': 'jparacrawl/',
                'base-ja-en': f'{_CT2_MODEL_FOLDERS["ja-en"]}',
                'base-en-ja': f'{_CT2_MODEL_FOLDERS["en-ja"]}',
            },
        },
    }

    # def _on_download_finished(self, map_key):
    #     print('Converting downloaded models to ct2 format')
    #     self._convert_fairseq_models_to_ct2(
    #         self._get_file_path(self._FAIRSEQ_MODEL_FILES['ja-en']),
    #         self._get_file_path('jparacrawl'),
    #         self._get_file_path(self._CT2_MODEL_FOLDERS['ja-en']),
    #         'ja', 'en',
    #     )
    #     self._convert_fairseq_models_to_ct2(
    #         self._get_file_path(self._FAIRSEQ_MODEL_FILES['en-ja']),
    #         self._get_file_path('jparacrawl'),
    #         self._get_file_path(self._CT2_MODEL_FOLDERS['en-ja']),
    #         'en', 'ja',
    #     )
    #     # os.remove(self._get_file_path(self._MODEL_FILES['en-ja']))
    #     # os.remove(self._get_file_path(self._MODEL_FILES['ja-en']))

    # def _convert_fairseq_models_to_ct2(self, model_path: str, data_dir: str, output_dir: str, from_lang: str, to_lang: str):
    #     cmds = [
    #         'ct2-fairseq-converter',
    #         '--model_path', model_path,
    #         '--data_dir', data_dir,
    #         '--output_dir', output_dir,
    #         '--source_lang', from_lang,
    #         '--target_lang', to_lang,
    #     ]
    #     subprocess.check_call(cmds)

    async def _load(self, from_lang: str, to_lang: str, device: str):
        if from_lang == 'auto':
            if to_lang == 'en':
                from_lang = 'ja'
            else:
                from_lang = 'en'

        self.load_params = {
            'from_lang': from_lang,
            'to_lang': to_lang,
            'device': device,
        }
        self.model = ctranslate2.Translator(
            model_path=self._get_file_path(self._CT2_MODEL_FOLDERS[f'{from_lang}-{to_lang}']),
            device=device,
            device_index=0,
        )
        self.model.load_model()
        self.sentence_piece_processors = {
            'en': spm.SentencePieceProcessor(model_file=self._get_file_path('jparacrawl/spm.en.nopretok.model')),
            'ja': spm.SentencePieceProcessor(model_file=self._get_file_path('jparacrawl/spm.ja.nopretok.model')),
        }

    async def _unload(self):
        self.model.unload_model()
        del self.model
        del self.sentence_piece_processors

    async def infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            if to_lang == 'en':
                from_lang = 'ja'
            else:
                from_lang = 'en'
        if self.is_loaded() and to_lang != self.load_params['to_lang']:
            await self.reload(self.load_params['device'], from_lang, to_lang)

        return await super().infer(from_lang, to_lang, queries)

    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        queries_tokenized = self.tokenize(queries, from_lang)
        translated_tokenized = self.model.translate_batch(
            source=queries_tokenized,
            beam_size=5,
            num_hypotheses=1,
            return_alternatives=False,
            disable_unk=True,
            replace_unknowns=True,
            repetition_penalty=3,
        )
        translated = self.detokenize(list(map(lambda t: t[0]['tokens'], translated_tokenized)), to_lang)
        return translated

    def tokenize(self, queries, lang):
        sp = self.sentence_piece_processors[lang]
        if isinstance(queries, list):
            return sp.encode(queries, out_type=str)
        else:
            return [sp.encode(queries, out_type=str)]

    def detokenize(self, queries, lang):
        sp = self.sentence_piece_processors[lang]
        translation = sp.decode(queries)
        return translation

class JparacrawlBigTranslator(JparacrawlTranslator):
    _CT2_MODEL_FOLDERS = {
        'ja-en': 'jparacrawl/big-ja-en',
        'en-ja': 'jparacrawl/big-en-ja',
    }
    _MODEL_MAPPING = {
        'models': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/jparacrawl-big-models.zip',
            'hash': '5e0c4cea5a5098152f566de3694602ed3db52927d3df22d2a7bfb8dba2bebe33',
            'archive': {
                'spm.ja.nopretok.model': 'jparacrawl/',
                'spm.en.nopretok.model': 'jparacrawl/',
                'big-ja-en': f'{_CT2_MODEL_FOLDERS["ja-en"]}',
                'big-en-ja': f'{_CT2_MODEL_FOLDERS["en-ja"]}',
            },
        },
    }

class SugoiTranslator(JparacrawlBigTranslator):
    """
    Sugoi model V4.0 for ja->en translation. For en->ja it falls back to jparacrawl.
    """
    _CT2_MODEL_FOLDERS = {
        'ja-en': 'sugoi/big-ja-en',
        'en-ja': 'jparacrawl/big-en-ja',
    }
    _MODEL_MAPPING = {
        **JparacrawlBigTranslator._MODEL_MAPPING,
        'models-sugoi': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/sugoi-models.zip',
            'hash': '67e060a62dc16211157a5eaa4fa8f72c86db5999fc69322606a6fcdf57f587f7',
            'archive': {
                'spm.ja.nopretok.model': 'sugoi/',
                'spm.en.nopretok.model': 'sugoi/',
                'big-ja-en': f'{_CT2_MODEL_FOLDERS["ja-en"]}',
            },
        },
    }

    def __init__(self):
        self.query_split_sizes = []
        super().__init__()

    async def _load(self, from_lang: str, to_lang: str, device: str):
        await super()._load(from_lang, to_lang, device)
        self.sentence_piece_processors['en-sugoi'] = spm.SentencePieceProcessor(model_file=self._get_file_path('sugoi/spm.en.nopretok.model'))
        self.sentence_piece_processors['ja-sugoi'] = spm.SentencePieceProcessor(model_file=self._get_file_path('sugoi/spm.ja.nopretok.model'))

    def tokenize(self, queries, lang):
        if lang == 'ja':
            lang = 'ja-sugoi'
            new_queries = []
            self.query_split_sizes = []
            for q in queries:
                # Split sentences into their own queries to prevent abbreviations
                # Add \s to make current sentence include whitespace after it...
                sentences = re.split(r'(\w[.‥…!?。・]+)', q)
                chunk_queries = []
                # Two sentences per query
                for chunk in chunks(sentences, 4):
                    chunk_queries.append(''.join(chunk).replace('.', '@'))
                self.query_split_sizes.append(len(chunk_queries))
                new_queries.extend(chunk_queries)
            queries = new_queries
        return super().tokenize(queries, lang)

    def detokenize(self, queries, lang):
        if lang == 'en':
            lang = 'en-sugoi'
        translations = super().detokenize(queries, lang)
        if lang == 'en-sugoi':
            new_translations = []
            i = 0
            for query_count in self.query_split_sizes:
                sentences = ''
                for sentence in translations[i:i+query_count]:
                    sentences += sentence + ('. ' if not sentence.endswith('.') else ' ')
                sentences = sentences.replace('@', '.').replace('▁', ' ').replace('<unk>', '')
                i += query_count
                new_translations.append(sentences)
            translations = new_translations
        return translations
