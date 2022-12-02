from typing import List
import unicodedata
import sentencepiece as spm

from .common import OfflineTranslator

class SugoiTranslator(OfflineTranslator):
    _LANGUAGE_CODE_MAP = {
        'JPN': 'ja',
        'ENG': 'en',
    }
    _MODEL_FILES = {
        'ja-en': 'base.pretrain.ja-en.pt',
        'en-ja': 'base.pretrain.en-ja.pt',
    }
    _MODEL_MAPPING = {
        'spm': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/spm_models/en-ja_spm.tar.gz',
            'hash': '12ee719799022b9ef102ce828209e53876112b52b4363dc277caca682b1b1d2e',
            'archive': {
                'enja_spm_models/spm.ja.nopretok.model': 'jparacrawl/',
                'enja_spm_models/spm.en.nopretok.model': 'jparacrawl/',
            },
        },
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/base.tar.gz',
            'hash': '73e09a50d07e1f443135178b67d1cc9710753c169cae44688e5e15a10950686a',
            'archive': {
                'base/dict.en.txt': 'jparacrawl/',
                'base/dict.ja.txt': 'jparacrawl/',
                'base/LICENSE': 'jparacrawl/',
                'base/base.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/base.tar.gz',
            'hash': '5c92d6d8776a7c6e5ca1162cfa1dd179c1edb410ce49aa6e97b2bddeef60ab6e',
            'archive': {
                'base/dict.en.txt': 'jparacrawl/',
                'base/dict.ja.txt': 'jparacrawl/',
                'base/LICENSE': 'jparacrawl/',
                'base/base.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }

    async def _load(self, from_lang: str, to_lang: str, device: str):
        from fairseq.models.transformer import TransformerModel

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
        self.model = TransformerModel.from_pretrained(
            self._get_file_path('jparacrawl/'),
            checkpoint_file=self._MODEL_FILES[f'{from_lang}-{to_lang}'],
            data_name_or_path=self._get_file_path('jparacrawl/'),
            source_lang=from_lang,
            target_lang=to_lang,
        )
        self.model.to(device)
        self.sentence_piece_processors = {
            'en': spm.SentencePieceProcessor(model_file=self._get_file_path('jparacrawl/spm.en.nopretok.model')),
            'ja': spm.SentencePieceProcessor(model_file=self._get_file_path('jparacrawl/spm.ja.nopretok.model')),
        }

    async def _unload(self):
        del self.model
        del self.sentence_piece_processors

    async def forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if from_lang == 'auto':
            if to_lang == 'en':
                from_lang = 'ja'
            else:
                from_lang = 'en'
        if self.is_loaded() and to_lang != self.load_params['to_lang']:
            await self.reload(self.load_params['device'], from_lang, to_lang)

        return await super().forward(from_lang, to_lang, queries)

    async def _forward(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        # return self._translate_sentence(from_lang, to_lang, ' # '.join(queries)).split(' # ')
        return [self._translate_sentence(from_lang, to_lang, query) for query in queries]

    def _translate_sentence(self, from_lang: str, to_lang: str, query: str) -> str:
        query = self._preprocess(from_lang, query)
        translated = self.model.translate(query)
        translated = self._postprocess(to_lang, translated)
        print(f'Sugoi Translation[{from_lang} -> {to_lang}] "{query}" -> "{translated}"')
        return translated

    def _preprocess(self, lang: str, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = text.strip()
        text = ' '.join(text.split())
        spp = self.sentence_piece_processors[lang]
        text = spp.encode(text, out_type = int)
        text = ' '.join([spp.IdToPiece(i) for i in text])
        return text

    def _postprocess(self, lang: str, text: str) -> str:
        spp = self.sentence_piece_processors[lang]
        text = spp.decode(text.split(' '))
        #text = ' '.join(text.split()).replace('▁', '').strip()
        return text

class SugoiSmallTranslator(SugoiTranslator):
    _MODEL_FILES = {
        'ja-en': 'small.pretrain.ja-en.pt',
        'en-ja': 'small.pretrain.en-ja.pt',
    }
    _MODEL_MAPPING = {
        **SugoiTranslator._MODEL_MAPPING,
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/small.tar.gz',
            'hash': '7136fe12841c626b105a9e588f858a8e0b76e451b19839457d7473ec705d12b3',
            'archive': {
                'small/dict.en.txt': 'jparacrawl/',
                'small/dict.ja.txt': 'jparacrawl/',
                'small/LICENSE': 'jparacrawl/',
                'small/small.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/small.tar.gz',
            'hash': '3b5b60f2a57ee1fc698c004b53c9483e8458082262c526f02843158be1271b4b',
            'archive': {
                'small/dict.en.txt': 'jparacrawl/',
                'small/dict.ja.txt': 'jparacrawl/',
                'small/LICENSE': 'jparacrawl/',
                'small/small.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }

class SugoiBigTranslator(SugoiTranslator):
    _MODEL_FILES = {
        'ja-en': 'big.pretrain.ja-en.pt',
        'en-ja': 'big.pretrain.en-ja.pt',
    }
    _MODEL_MAPPING = {
        **SugoiTranslator._MODEL_MAPPING,
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/big.tar.gz',
            'hash': '7517753b6feb8594d3c86ad7742dbc49203115add21e8a6c7542aa2ac0df1c6a',
            'archive': {
                'big/dict.en.txt': 'jparacrawl/',
                'big/dict.ja.txt': 'jparacrawl/',
                'big/LICENSE': 'jparacrawl/',
                'big/big.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/big.tar.gz',
            'hash': '520cd7c2b4b84c3fbb5a7a948e5183b7bca4dc551f91268f0e2dbeb98cb8b77d',
            'archive': {
                'big/dict.en.txt': 'jparacrawl/',
                'big/dict.ja.txt': 'jparacrawl/',
                'big/LICENSE': 'jparacrawl/',
                'big/big.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }
