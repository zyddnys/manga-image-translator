import os
import sys
import pytest
from typing import List
import numpy as np

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.translators import dispatch as dispatch_translation

@pytest.mark.asyncio
async def test_mixed_languages():
    queries = ['How to be dead everyday', '', 'Ich bin ein deutscher', '我想每天学习如何变得更同性恋', 'HELLO THERE I WANT an audition!', '目标意识']
    translator = 'google'
    print(await dispatch_translation(translator, 'auto', 'RUS', queries))
    raise Exception()
    
@pytest.mark.asyncio
async def test_single_language():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    translator = 'google'
    print(await dispatch_translation(translator, 'auto', 'ENG', queries))
    raise Exception()
