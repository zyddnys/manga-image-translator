import base64
import os
from io import BytesIO

import cv2
import httpx
import numpy as np
from PIL import Image

from ..utils.log import get_logger

logger = get_logger("MiniCPM-OCR")


def _line_colors(crop):
    # Estimate paper from the border and ink from the strongest interior strokes.
    pixels = crop[1:-1, 1:-1] if min(crop.shape[:2]) > 3 else crop
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    paper = np.median(border, axis=0).astype(np.int16)
    contrast = np.abs(pixels.astype(np.int16) - paper).max(axis=2).astype(np.uint8)
    _, strokes = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink = paper
    mask = strokes != 0
    if mask.any():
        core = mask & (contrast >= np.percentile(contrast[mask], 80))
        ink = np.median(pixels[core], axis=0)
    return tuple(map(int, ink)), tuple(map(int, paper))


async def _read_line(client, url, model, crop):
    with BytesIO() as output:
        Image.fromarray(crop).save(output, format="PNG")
        payload = base64.b64encode(output.getvalue()).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
        {"type": "text", "text": "Text Recognition:"},
    ]
    response = await client.post(url, json={
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 512,
    })
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice["finish_reason"] != "stop":
        return ""
    text = choice["message"]["content"]
    return text.strip() if isinstance(text, str) else ""


async def recover_missing_lines(image, missing):
    """Retry boxes rejected by 48px, using the directions assigned on the full page."""
    base_url = os.environ.get("MINICPM_OCR_BASE_URL")
    if not base_url:
        logger.warning("Set MINICPM_OCR_BASE_URL to enable fallback for missed lines.")
        return []
    url = base_url.rstrip("/") + "/chat/completions"
    model = os.environ.get("MINICPM_OCR_MODEL", "minicpm-ocr")
    api_key = os.environ.get("MINICPM_OCR_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    recovered = []
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        for line in missing:
            direction = line.assigned_direction
            crop = line.get_transformed_region(image, direction, max(2, round(line.font_size)))
            if direction == "v":
                # Undo 48px's rotation so the service sees upright vertical text.
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            try:
                text = await _read_line(client, url, model, crop)
            except (httpx.RequestError, httpx.InvalidURL) as exc:
                logger.warning("MiniCPM-OCR unavailable (%s), skipping remaining lines.", type(exc).__name__)
                break
            except httpx.HTTPStatusError as exc:
                logger.warning("MiniCPM-OCR returned HTTP %s.", exc.response.status_code)
                if exc.response.status_code in (401, 403, 404, 429):
                    break
                continue
            except (ValueError, KeyError, IndexError, TypeError):
                logger.warning("MiniCPM-OCR returned an invalid completion.")
                continue
            if text:
                fg, bg = _line_colors(crop)
                line.fg_r, line.fg_g, line.fg_b = fg
                line.bg_r, line.bg_g, line.bg_b = bg
                line.text = text
                recovered.append(line)
    return recovered
