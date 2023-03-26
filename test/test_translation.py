import os
import sys
import pytest

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.translators import (
    TRANSLATORS,
    TranslatorChain,
    OfflineTranslator,
    MissingAPIKeyException,
    dispatch,
)

@pytest.mark.asyncio
async def test_mixed_languages():
    queries = ['How to be dead everyday', '', 'Ich bin ein deutscher', 'Test case m. HELLO THERE I WANT an audition! YOYOYOYO', '目标意识']
    chain = TranslatorChain('google:ENG')
    print(await dispatch(chain, queries))
    
@pytest.mark.asyncio
async def test_single_language():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    chain = TranslatorChain('google:ENG')
    print(await dispatch(chain, queries))
    
@pytest.mark.asyncio
async def test_chain():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    chain = TranslatorChain('google:JPN;sugoi:ENG')
    print(await dispatch(chain, queries))

@pytest.mark.asyncio
async def test_online_translators():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    for key in TRANSLATORS:
        if issubclass(TRANSLATORS[key], OfflineTranslator):
            continue
        try:
            chain = TranslatorChain(f'{key}:ENG')
            print(await dispatch(chain, queries))
        except MissingAPIKeyException as e:
            print(e)

@pytest.mark.asyncio
async def test_offline_translators():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    for key in ('offline', 'sugoi', 'm2m100'):
        if not issubclass(TRANSLATORS[key], OfflineTranslator):
            continue
        chain = TranslatorChain(f'{key}:ENG')
        print(await dispatch(chain, queries))
