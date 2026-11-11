"""API Route regressions with the real OpenAI SDK and mocked HTTP responses."""

import asyncio
import json
import runpy
from pathlib import Path

import httpx
import openai
import pytest
import pytest_asyncio

from manga_translator.config import Translator
from manga_translator.translators import api_route, keys
from manga_translator.translators.common import MissingAPIKeyException


@pytest_asyncio.fixture
async def make_translator(monkeypatch):
    monkeypatch.setattr(api_route, 'API_ROUTE_API_KEY', 'test-api-route-key')
    monkeypatch.setattr(api_route, 'API_ROUTE_API_BASE', 'https://api-route.invalid/v1')
    monkeypatch.setattr(api_route, 'API_ROUTE_MODEL', 'gpt-6.1-sol')
    clients = []

    def make(replies):
        requests = []
        remaining = iter(replies)

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                'id': 'test-completion', 'object': 'chat.completion', 'created': 0,
                'model': 'gpt-6.1-sol',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': next(remaining)}}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12},
            })

        client = openai.AsyncOpenAI(
            api_key='test-api-route-key', base_url='https://api-route.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        def build_client(**kwargs):
            assert kwargs == {'api_key': 'test-api-route-key', 'base_url': 'https://api-route.invalid/v1'}
            return client

        with monkeypatch.context() as patch:
            patch.setattr(api_route.openai, 'AsyncOpenAI', build_client)
            translator = api_route.ApiRouteTranslator()
        translator._MAX_REQUESTS_PER_MINUTE = 0
        return translator, requests

    yield make
    for client in clients:
        await client.close()


def test_api_route_config_and_enum(monkeypatch):
    monkeypatch.setenv('API_ROUTE_API_KEY', 'configured-key')
    monkeypatch.setenv('API_ROUTE_API_BASE', 'https://custom.invalid/v1')
    monkeypatch.setenv('API_ROUTE_MODEL', 'other/model')
    config = runpy.run_path(str(Path(keys.__file__)))
    assert config['API_ROUTE_API_KEY'] == 'configured-key'
    assert config['API_ROUTE_API_BASE'] == 'https://custom.invalid/v1'
    assert config['API_ROUTE_MODEL'] == 'other/model'
    assert Translator('api_route') == Translator.api_route


def test_api_route_config_defaults(monkeypatch):
    for variable in ('API_ROUTE_API_KEY', 'API_ROUTE_API_BASE', 'API_ROUTE_MODEL'):
        monkeypatch.delenv(variable, raising=False)
    config = runpy.run_path(str(Path(keys.__file__)))
    assert config['API_ROUTE_API_KEY'] == ''
    assert config['API_ROUTE_API_BASE'] == 'https://global.api-route.com/v1'
    assert config['API_ROUTE_MODEL'] == 'gpt-6.1-sol'


def test_api_route_requires_key(monkeypatch):
    monkeypatch.setattr(api_route, 'API_ROUTE_API_KEY', '')
    with pytest.raises(MissingAPIKeyException, match='API_ROUTE_API_KEY'):
        api_route.ApiRouteTranslator()


@pytest.mark.asyncio
async def test_translation_preserves_order_and_uses_configured_transport(make_translator):
    translator, requests = make_translator(['<|1|>Hello!\n<|2|>Goodbye.'])
    assert await translator._translate('Japanese', 'English', ['こんにちは！', 'さようなら。']) == ['Hello!', 'Goodbye.']
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == 'https://api-route.invalid/v1/chat/completions'
    assert request.headers['Authorization'] == 'Bearer test-api-route-key'
    assert payload['model'] == 'gpt-6.1-sol'
    assert '<|1|>こんにちは！' in payload['messages'][-1]['content']
    assert payload['max_tokens'] > 0
    assert translator.token_count == 12
    assert translator.count_tokens('中文') == len('中文'.encode('utf-8'))


@pytest.mark.asyncio
async def test_translation_retries_a_missing_segment_marker(make_translator):
    translator, requests = make_translator(['Hello!', '<|1|>Hello!'])
    assert await translator._translate('Japanese', 'English', ['こんにちは！']) == ['Hello!']
    assert len(requests) == 2
    assert translator.token_count == 24


@pytest.mark.asyncio
async def test_timeout_cancels_and_awaits_each_request(make_translator, monkeypatch):
    translator, _ = make_translator([])
    translator._TIMEOUT = 0.01
    translator._TIMEOUT_RETRY_ATTEMPTS = 1
    cancelled = []

    async def slow_request(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(translator, '_request_translation', slow_request)
    with pytest.raises(asyncio.TimeoutError, match='API Route'):
        await translator._request_with_timeout('English', 'hello')
    assert len(cancelled) == 2
