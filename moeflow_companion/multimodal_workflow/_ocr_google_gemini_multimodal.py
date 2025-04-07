from google import genai
from pathlib import Path
from pydantic import BaseModel, Field
from ..llm_clients import gemini_bare

import logging

logger = logging.getLogger(__name__)


class TextRange(BaseModel):
    left: float = Field(description="左端")
    top: float = Field(description="上端")
    width: float = Field(description="幅")
    height: float = Field(description="高さ")
    text: str = Field(description="原文")
    translation: str = Field(description="訳文")


class ImageProcessResult(BaseModel):
    items: list[TextRange]


def process_images(
    image_files: list[Path],
    target_lang: str,
    model: gemini_bare.GcpGeminiBare | str = gemini_bare.GcpGeminiBare.gemini25_pro_exp,
) -> list[ImageProcessResult]:
    model = gemini_bare.GcpGeminiBare(model) if isinstance(model, str) else model
    assert isinstance(model, gemini_bare.GcpGeminiBare), (
        f"model should be of type {gemini_bare.GcpGeminiBare}, but got {type(model)}"
    )
    return [
        _process_single_image(
            i,
            target_lang=target_lang,
            client=gemini_bare.get_gemini_client(),
            model=model,
        )
        for i in image_files
    ]


def _process_single_image(
    image_file: Path,
    *,
    target_lang: str,
    client: genai.Client,
    model: gemini_bare.GcpGeminiBare,
) -> ImageProcessResult:
    prompt = f"""
添付の漫画ページから、文字を抽出してください。また {target_lang} に翻訳を行ってください。
注意：
- 余計な説明は不要です
- テキストの内容のみを抽出してください
- 複数行に分かれたテキストを一件にまとめてください
- 吹き出し、手書き文字、音喩などを全部含めてください
"""
    response = model.complete_with_json(
        user_messages=[image_file, prompt], res_model=ImageProcessResult, client=client
    )
    logger.info("processed image %s", image_file)
    return response
