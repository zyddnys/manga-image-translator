"""_summary_"""

from pathlib import Path
from pydantic import BaseModel, Field


class MoeflowProjectMeta(BaseModel):
    name: str
    intro: str
    output_language: str = "en"
    default_role: str = "supporter"
    allow_apply_type: int = 3
    application_check_type: int = 1
    is_need_check_application: bool = False
    source_language: str = "ja"


class MoeflowTextBlock(BaseModel):
    pass


class MoeflowFile(BaseModel):
    local_file: Path
    image_w: int
    image_h: int
    text_blocks: list[MoeflowTextBlock]


class MoeflowProject(BaseModel):
    meta: MoeflowProjectMeta
    files: list[MoeflowFile]
