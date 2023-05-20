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
async def test_specified_translator(translator, tgt_lang, text, count):
    if translator is None:
        pytest.skip()

    print()
    for i in range(count):
        if text is None:
            queries_list = [
                ['Hallo', '', 'English is a West Germanic language in the Indo-European language family, with its earliest forms spoken by the inhabitants of early medieval England.', 'Test case 5. HELLO THERE I WANT an audition! YOYOYOYO', '目标意识'],
                ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。'],
                ['....DO YOU HAVE EXPERIENCE IN TAKING CARE OF SICK PEOPLE..?'],
            ]
        else:
            queries_list = [[text]]
        

        chain = TranslatorChain(f'{translator}:{tgt_lang}')
        for queries in queries_list:
            print(queries)
            print('-->')
            print(await dispatch(chain, queries))
            print()
