from typing import Iterable
from google import genai
from pathlib import Path
from pydantic import BaseModel, Field
from ..llm_clients import gemini_bare
import asyncio
import contextlib

import logging

logger = logging.getLogger(__name__)

# A primitive limiter: no more than 5 concurrent reqs.
# NOTE The real limit is more complicated and contains per-model RPM TPM RPD
# see https://ai.google.dev/gemini-api/docs/rate-limits#tier-1
global_semaphore = asyncio.Semaphore(5)


logger.setLevel(logging.INFO)


class TextRange(BaseModel):
    left: float = Field(description="left X of text")
    top: float = Field(description="top Y of text")
    right: float = Field(description="right X of txt")
    bottom: float = Field(description="bottom Y of text")
    text: str = Field(description="the original text")
    translated_text: str = Field(description="the translated text")


class ImageProcessResult(BaseModel):
    items: list[TextRange]


async def process_images(
    image_files: list[Path],
    *,
    target_lang: str,
    model: gemini_bare.GcpGeminiBare | str = gemini_bare.GcpGeminiBare.gemini25_flash,
) -> list[ImageProcessResult]:
    model = gemini_bare.GcpGeminiBare(model) if isinstance(model, str) else model
    assert isinstance(model, gemini_bare.GcpGeminiBare), (
        f"model should be of type {gemini_bare.GcpGeminiBare}, but got {type(model)}"
    )
    return await asyncio.gather(
        *[
            _process_single_image(
                i,
                target_lang=target_lang,
                client=gemini_bare.get_gemini_client(),
                model=model,
                semaphores=(global_semaphore,),
            )
            for i in image_files
        ]
    )


async def _process_single_image(
    image_file: Path,
    *,
    target_lang: str,
    client: genai.Client,
    model: gemini_bare.GcpGeminiBare,
    semaphores: Iterable[contextlib.AbstractContextManager] = (),
) -> ImageProcessResult:
    prompt = f"""
Please extract text from the attached manga page. Also, translate it into {target_lang}.

Your instructions are as follows:

- Please only extract the text content, do not include other information
- If text is in multiple lines, please merge them into one line.
- Please include all texts, including speech bubbles, hand-written text, sound effects, and other you can recocgnize.
- All x y should be in [0, 1000] normalized coordinates.
"""
    async with contextlib.AsyncExitStack() as stack:
        for s in semaphores:
            await stack.enter_async_context(s)
        logger.info("processing image %s", image_file)

        response = await model.complete_with_json_async(
            user_messages=[image_file, prompt],
            res_model=ImageProcessResult,
            client=client,
        )
        logger.info("processed image %s", image_file)
        for i, item in enumerate(response.items):
            logger.info("item %d: %s", i, item)
        return response
