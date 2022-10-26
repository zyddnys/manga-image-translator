import os
from typing import List
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel

from .common import OfflineTranslator

class JParaCrawlTranslator(OfflineTranslator):
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
            'hash': '12EE719799022B9EF102CE828209E53876112B52B4363DC277CACA682B1B1D2E',
            'archive-files': {
                'enja_spm_models/spm.ja.nopretok.model': 'jparacrawl/',
                'enja_spm_models/spm.en.nopretok.model': 'jparacrawl/',
            },
        },
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/base.tar.gz',
            'hash': '73E09A50D07E1F443135178B67D1CC9710753C169CAE44688E5E15A10950686A',
            'archive-files': {
                'base/dict.en.txt': 'jparacrawl/',
                'base/dict.ja.txt': 'jparacrawl/',
                'base/LICENSE': 'jparacrawl/',
                'base/base.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/base.tar.gz',
            'hash': '5C92D6D8776A7C6E5CA1162CFA1DD179C1EDB410CE49AA6E97B2BDDEEF60AB6E',
            'archive-files': {
                'base/dict.en.txt': 'jparacrawl/',
                'base/dict.ja.txt': 'jparacrawl/',
                'base/LICENSE': 'jparacrawl/',
                'base/base.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }

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
        print('Loading model')

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
        if self.is_loaded() and (from_lang != self.load_params['from_lang'] or to_lang != self.load_params['to_lang']):
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

class JParaCrawlSmallTranslator(JParaCrawlTranslator):
    _MODEL_FILES = {
        'ja-en': 'small.pretrain.ja-en.pt',
        'en-ja': 'small.pretrain.en-ja.pt',
    }
    _MODEL_MAPPING = {
        **JParaCrawlTranslator._MODEL_MAPPING,
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/small.tar.gz',
            'hash': '7136FE12841C626B105A9E588F858A8E0B76E451B19839457D7473EC705D12B3',
            'archive-files': {
                'small/dict.en.txt': 'jparacrawl/',
                'small/dict.ja.txt': 'jparacrawl/',
                'small/LICENSE': 'jparacrawl/',
                'small/small.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/small.tar.gz',
            'hash': '3B5B60F2A57EE1FC698C004B53C9483E8458082262C526F02843158BE1271B4B',
            'archive-files': {
                'small/dict.en.txt': 'jparacrawl/',
                'small/dict.ja.txt': 'jparacrawl/',
                'small/LICENSE': 'jparacrawl/',
                'small/small.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }

class JParaCrawlBigTranslator(JParaCrawlTranslator):
    _MODEL_FILES = {
        'ja-en': 'big.pretrain.ja-en.pt',
        'en-ja': 'big.pretrain.en-ja.pt',
    }
    _MODEL_MAPPING = {
        **JParaCrawlTranslator._MODEL_MAPPING,
        'model-ja-en': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/big.tar.gz',
            'hash': '7517753B6FEB8594D3C86AD7742DBC49203115ADD21E8A6C7542AA2AC0DF1C6A',
            'archive-files': {
                'big/dict.en.txt': 'jparacrawl/',
                'big/dict.ja.txt': 'jparacrawl/',
                'big/LICENSE': 'jparacrawl/',
                'big/big.pretrain.pt': f'jparacrawl/{_MODEL_FILES["ja-en"]}',
            },
        },
        'model-en-ja': {
            'url': 'http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/big.tar.gz',
            'hash': '520CD7C2B4B84C3FBB5A7A948E5183B7BCA4DC551F91268F0E2DBEB98CB8B77D',
            'archive-files': {
                'big/dict.en.txt': 'jparacrawl/',
                'big/dict.ja.txt': 'jparacrawl/',
                'big/LICENSE': 'jparacrawl/',
                'big/big.pretrain.pt': f'jparacrawl/{_MODEL_FILES["en-ja"]}',
            },
        },
    }
