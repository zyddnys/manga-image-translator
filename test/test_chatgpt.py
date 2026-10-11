"""ChatGPT regression tests using the real OpenAI SDK with a mocked HTTP transport."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
import pytest_asyncio
from omegaconf import OmegaConf

from manga_translator.config import Translator, TranslatorChain
from manga_translator.translators import chatgpt, chatgpt_2stage, keys


@pytest_asyncio.fixture
async def make_translator(monkeypatch, tmp_path):
    monkeypatch.setattr(chatgpt, 'OPENAI_API_KEY', 'test-key')
    monkeypatch.setattr(chatgpt, 'OPENAI_GLOSSARY_PATH', str(tmp_path / 'missing.txt'))
    monkeypatch.setattr(chatgpt, 'OPENAI_HTTP_PROXY', None)
    for variable in ('OPENAI_MODEL', 'OPENAI_STAGE1_MODEL', 'OPENAI_STAGE2_MODEL', 'OPENAI_FALLBACK_MODEL'):
        monkeypatch.setenv(variable, 'unsupported-model')
    clients = []

    def make(replies, translator_type=chatgpt.OpenAITranslator):
        requests = []
        remaining = iter(replies)

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            assert payload['model'] == 'gpt-6-luna'
            assert payload['reasoning_effort'] == 'none'
            assert payload['max_completion_tokens'] > 0
            assert 'max_tokens' not in payload
            reply = next(remaining)
            return httpx.Response(200, json={
                'id': 'test-completion', 'object': 'chat.completion', 'created': 0,
                'model': 'gpt-6-luna',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': reply}}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12},
            })

        client = openai.AsyncOpenAI(
            api_key='test-key', base_url='https://openai.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        with monkeypatch.context() as patch:
            patch.setattr(chatgpt.openai, 'AsyncOpenAI', lambda **kwargs: client)
            translator = translator_type()
        translator._MAX_REQUESTS_PER_MINUTE = 0
        translator.print_boxed = lambda *args, **kwargs: None
        return translator, requests

    yield make
    for client in clients:
        await client.close()


@pytest.mark.parametrize('alias', ['chatgpt', 'openai', 'gpt-6-luna'])
def test_supported_aliases(alias):
    assert Translator(alias) == Translator.chatgpt
    assert TranslatorChain(f'{alias}:ENG').translators == (Translator.chatgpt,)


@pytest.mark.parametrize('alias', ['gpt-4o', 'gpt-3.5-turbo', 'gpt-unknown'])
def test_old_model_aliases_are_rejected(alias):
    with pytest.raises(ValueError):
        Translator(alias)


def test_model_cannot_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL', 'unsupported-model')
    assert runpy.run_path(str(Path(keys.__file__)))['OPENAI_MODEL'] == 'gpt-6-luna'


@pytest.mark.asyncio
async def test_translation_preserves_config_glossary_and_context(make_translator):
    translator, requests = make_translator(['<|1|>Hello, Luna!\n<|2|>Goodbye.'])
    translator.config = OmegaConf.create({'chatgpt': {'gpt-6-luna': {'temperature': 0.4, 'top_p': 0.8}}})
    translator.glossary_entries = {'ルナ': 'Luna'}
    translator.set_prev_context('The speaker is Luna.')

    assert await translator._translate('JPN', 'ENG', ['こんにちは、ルナ！', 'さようなら。']) == ['Hello, Luna!', 'Goodbye.']
    payload = requests[0]
    assert payload['temperature'] == 0.4
    assert payload['top_p'] == 0.8
    assert any('ルナ->Luna' in m['content'] for m in payload['messages'])
    assert any(m['content'] == 'The speaker is Luna.' for m in payload['messages'])
    assert translator.token_count == 12


@pytest.mark.asyncio
async def test_invalid_batch_splits_using_same_model(make_translator):
    translator, requests = make_translator(['invalid format', '<|1|>One', '<|1|>Two'])
    translator._RETRY_ATTEMPTS = 0

    assert await translator._translate('JPN', 'ENG', ['一', '二']) == ['One', 'Two']
    assert len(requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('translator_type', [chatgpt.OpenAITranslator, chatgpt_2stage.ChatGPT2StageTranslator])
async def test_empty_response_is_reported(make_translator, translator_type):
    translator, _ = make_translator([None], translator_type)
    with pytest.raises(ValueError, match='Empty translation content'):
        await translator._request_translation('ENG', '<|1|>こんにちは')


def page(texts):
    return SimpleNamespace(
        img_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        text_regions=[SimpleNamespace(text=text, xyxy=(0, 0, 16, 16)) for text in texts],
        verbose=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('batch', [False, True])
async def test_ocr_and_translation_use_luna_and_preserve_reading_order(make_translator, batch):
    regions = [
        {'reading_order': 0, 'original_bbox_id': 1, 'corrected_text': '二'},
        {'reading_order': 1, 'original_bbox_id': 0, 'corrected_text': '一'},
    ]
    refinement = {'corrected_regions': regions, 'image_received': True}
    if batch:
        refinement = {'batch_results': [{'image_index': 0, 'corrected_regions': regions}]}
    translator, requests = make_translator(
        [json.dumps(refinement), '<|1|>Two\n<|2|>One'], chatgpt_2stage.ChatGPT2StageTranslator,
    )
    context = page(['一', '二'])
    if batch:
        context.batch_contexts = [page(['一']), page(['二'])]

    assert await translator._translate('JPN', 'ENG', ['一', '二'], context) == ['One', 'Two']
    assert len(requests) == 2
    assert requests[0]['response_format']['type'] == 'json_schema'
    for payload in requests:
        content = payload['messages'][-1]['content']
        images = [item for item in content if item['type'] == 'image_url']
        assert len(images) == (2 if batch else 1)
    assert not translator._is_stage2_translation


@pytest.mark.asyncio
async def test_ocr_failure_keeps_original_text_without_switching_model(make_translator):
    translator, requests = make_translator(
        ['I cannot assist with that.', '<|1|>One'], chatgpt_2stage.ChatGPT2StageTranslator,
    )
    assert await translator._translate('JPN', 'ENG', ['一'], page(['一'])) == ['One']
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('proxy', ['localhost:8080', 'http://localhost:8080'])
async def test_httpx_proxy_configuration(monkeypatch, tmp_path, proxy):
    monkeypatch.setattr(chatgpt, 'OPENAI_API_KEY', 'test-key')
    monkeypatch.setattr(chatgpt, 'OPENAI_HTTP_PROXY', proxy)
    monkeypatch.setattr(chatgpt, 'OPENAI_GLOSSARY_PATH', str(tmp_path / 'missing.txt'))
    translator = chatgpt.OpenAITranslator()
    try:
        client = translator.client._client
        proxied = client._transport_for_url(httpx.URL('https://api.openai.com/v1'))
        direct = client._transport_for_url(httpx.URL('https://example.invalid/v1'))
        assert proxied is not direct
    finally:
        await translator.client.close()
