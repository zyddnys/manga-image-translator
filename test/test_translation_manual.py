import os
import sys
import pytest

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.translators import (
    TranslatorChain,
    dispatch,
)

@pytest.mark.asyncio
async def test_specified_translator(translator, tgt_lang):
    if translator is None:
        pytest.skip()

    queries_list = [
        ['How to be dead everyday', '', 'Ich bin ein deutscher', '我想每天学习如何变得更同性恋', 'HELLO THERE I WANT an audition!', '目标意识'],
        ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。'],
    ]
    chain = TranslatorChain(f'{translator}:{tgt_lang}')
    for queries in queries_list:
        print(await dispatch(chain, queries))
