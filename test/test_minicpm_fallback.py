import json
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest

from manga_translator.config import OcrConfig
from manga_translator.ocr.minicpm import recover_missing_lines
from manga_translator.ocr.model_48px import Model48pxOCR
from manga_translator.utils import Quadrilateral


@pytest.fixture
def lines_and_image():
    image = np.full((60, 48, 3), 255, dtype=np.uint8)
    lines = []
    for top in (2, 20, 38):
        points = np.array([[2, top], [42, top], [42, top + 14], [2, top + 14]])
        line = Quadrilateral(points, "", 0.75)
        line.assigned_direction = "h"
        lines.append(line)
        image[top + 4:top + 10, 9:30] = 0
    return lines, image


def test_fallback_is_opt_in():
    assert OcrConfig().minicpm_fallback is False


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_48px_retries_only_rejected_boxes(monkeypatch, tmp_path, lines_and_image, enabled):
    lines, image = lines_and_image
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Model48pxOCR, "_MODEL_DIR", str(tmp_path))
    ocr = Model48pxOCR()
    ocr.use_gpu = False
    ocr.model = Mock(dictionary=["甲", "乙", "丙"])
    rgb = np.zeros((1, 3))
    present = np.array([[0, 1]])
    ocr.model.infer_beam_batch_tensor.return_value = [
        ([token], score, rgb, rgb, present, present)
        for token, score in enumerate([0.9, 0.4, 0.8])
    ]
    recover = AsyncMock(return_value=[lines[1]])
    monkeypatch.setattr("manga_translator.ocr.minicpm.recover_missing_lines", recover)

    result = await ocr._infer(image, lines, OcrConfig(prob=0.5, minicpm_fallback=enabled))

    assert result[:2] == [lines[0], lines[2]]
    assert [(line.text, line.prob) for line in result[:2]] == [("甲", 0.9), ("丙", 0.8)]
    if enabled:
        recover.assert_awaited_once_with(image, [lines[1]])
        assert result[2:] == [lines[1]]
    else:
        recover.assert_not_awaited()
        assert result[2:] == []


@pytest.mark.asyncio
async def test_only_complete_nonempty_replies_recover_lines(monkeypatch, lines_and_image):
    lines, image = lines_and_image
    monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://ocr.invalid/v1")
    replies = iter([("stop", " 完 "), ("length", "截断"), ("stop", "   ")])
    requests = []

    async def send(self, request, **kwargs):
        requests.append(request)
        finish, text = next(replies)
        choice = {"finish_reason": finish, "message": {"content": text}}
        return httpx.Response(200, json={"choices": [choice]}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    recovered = await recover_missing_lines(image, lines)

    assert recovered == [lines[0]]
    assert [line.text for line in lines] == ["完", "", ""]
    assert lines[0].prob == 0.75
    assert len(requests) == 3
    for request in requests:
        assert request.url == httpx.URL("http://ocr.invalid/v1/chat/completions")
        content = json.loads(request.content)["messages"][0]["content"]
        assert any(part.get("text") == "Text Recognition:" for part in content)
        assert any(part.get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
                   for part in content)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False], ids=["connection-refused", "no-endpoint"])
async def test_unavailable_service_leaves_line_untouched(monkeypatch, lines_and_image, configured):
    lines, image = lines_and_image
    if configured:
        monkeypatch.setenv("MINICPM_OCR_BASE_URL", "http://ocr.invalid/v1")
    else:
        monkeypatch.delenv("MINICPM_OCR_BASE_URL", raising=False)
    requests = []

    async def send(self, request, **kwargs):
        requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    assert await recover_missing_lines(image, lines[:1]) == []
    assert len(requests) == int(configured)
    assert lines[0].text == ""
    assert lines[0].prob == 0.75
