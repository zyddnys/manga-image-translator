import pytest

from manga_translator.translators import (
    TRANSLATORS,
    TranslatorChain,
    OfflineTranslator,
    MissingAPIKeyException,
    dispatch,
)
from manga_translator.translators.common import LanguageUnsupportedException

@pytest.mark.asyncio
async def test_mixed_languages():
    queries = ['How to be dead everyday', '', 'Ich bin ein deutscher', 'Test case m. HELLO THERE I WANT an audition! YOYOYOYO', '目标意识']
    try:
        chain = TranslatorChain('youdao:ENG')
        print(await dispatch(chain, queries))
    except MissingAPIKeyException as e:
        print(e)
    
@pytest.mark.asyncio
async def test_single_language():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    try:
        chain = TranslatorChain('youdao:CHS')
        print(await dispatch(chain, queries))
    except MissingAPIKeyException as e:
        print(e)
    
@pytest.mark.asyncio
async def test_chain():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    try:
        chain = TranslatorChain('gpt3.5:JPN;sugoi:ENG')
        print(await dispatch(chain, queries))
    except MissingAPIKeyException as e:
        print(e)

@pytest.mark.asyncio
async def test_online_translators():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    for key in TRANSLATORS:
        if issubclass(TRANSLATORS[key], OfflineTranslator):
            continue
        try:
            chain = TranslatorChain(f'{key}:ENG')
            print(await dispatch(chain, queries))
        except (MissingAPIKeyException, LanguageUnsupportedException) as e:
            print(e)

@pytest.mark.asyncio
async def test_offline_translators():
    queries = ['僕はアイネと共に一度、宿の方に戻った', '改めて直面するのは部屋の問題――部屋のベッドが一つでは、さすがに狭すぎるだろう。']
    for key in ('offline', 'sugoi', 'm2m100_big'):
        if not issubclass(TRANSLATORS[key], OfflineTranslator):
            continue
        chain = TranslatorChain(f'{key}:ENG')
        print(await dispatch(chain, queries))
