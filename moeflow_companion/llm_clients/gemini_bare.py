from enum import StrEnum
import functools
import os
from pathlib import Path
from google import genai
from google.genai import types as genai_types
import mimetypes

from typing import TypeVar
from pydantic import BaseModel

ModelClass = TypeVar("ModelClass", bound=BaseModel)


class GcpGeminiBare(StrEnum):
    # list of models (Gemini) : https://ai.google.dev/gemini-api/docs/models
    # list of models (Vertex AI):
    gemini25_pro_exp = "gemini-2.5-pro-exp-03-25"
    gemini20_flash = "gemini-2.0-flash"
    gemini20_flash_lite = "gemini-2.0-flash-lite"
    gemini15_flash = "gemini-1.5-flash"
    gemini15_flash_8b = "gemini-1.5-flash-8b"

    @classmethod
    def upload_file(
        kls, file_path: str | Path, *, client: genai.Client | None = None
    ) -> genai_types.File:
        if not client:
            client = get_gemini_client()
        return client.files.upload(file_path)

    def complete(
        self,
        *,
        user_messages: list[str | genai_types.File | Path],
        client: genai.Client | None = None,
        **kwargs,
    ) -> str:
        if not client:
            client = get_gemini_client()

        contents: list[genai_types.Part] = _build_parts(user_messages)

        response = client.models.generate_content(
            model=self.value,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=kwargs.get("temperature", 0.1)
            ),
        )
        return response.text

    def complete_with_json(
        self,
        *,
        user_messages: list[str | genai_types.File | Path],
        res_model: type[ModelClass],
        client: genai.Client | None = None,
        # *,
        # **kwargs,
    ) -> ModelClass:
        if not client:
            client = get_gemini_client()

        contents: list[genai_types.Part] = _build_parts(user_messages)

        response = client.models.generate_content(
            model=self.value,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=res_model,
            ),
            contents=contents,
        )
        return res_model.model_validate_json(response.text)


@functools.lru_cache(maxsize=1)
def get_gemini_client():
    # auth via Gemini API key
    key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not key:
        raise ValueError("Please set the GOOGLE_GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=key)

    # auth via GCP Vertex AI API
    # return genai.Client(vertexai=True, project="PROJECT_ID", location="us-central1")


@functools.lru_cache(maxsize=16)
def get_vertexai_client(project: str, location: str) -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


def _build_parts(
    user_messages: list[str | genai_types.File | Path],
) -> list[genai_types.Part]:
    contents: list[genai_types.Part] = []

    for s in user_messages:
        if isinstance(s, Path):
            file_bytes = s.read_bytes()
            mime_type = mimetypes.guess_type(s)[0]
            contents.append(
                genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            )
        elif isinstance(s, str):
            contents.append(s)
        elif isinstance(s, genai_types.File):
            contents.append(s)
        else:
            raise ValueError(f"Unsupported user_message type: {type(s)}")
    return contents
