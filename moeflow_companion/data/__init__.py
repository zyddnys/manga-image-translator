"""_summary_"""

from pathlib import Path
from pydantic import BaseModel, field_validator
import zipfile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    center_x: float  # [0..image_w]
    center_y: float  # [0..image_h]
    normalized_center_x: float  # [0..1]
    normalized_center_y: float  # [0..1]
    source: str | None
    translated: str | None

    @field_validator("normalized_center_x", "normalized_center_y", mode="after")
    @classmethod
    def validate_normalized_coord(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("must be a number")
        if not (0 <= v):
            # raise ValueError("must be between 0 and 1")
            logger.warning("normalized coord too small: %s using 0", v)
            return 0.0
        elif not (v <= 1):
            # raise ValueError("must be between 0 and 1")
            logger.warning("normalized coord too large: %s using 1", v)
            return 1.0

        return v


class MoeflowFile(BaseModel):
    local_path: Path
    image_w: int
    image_h: int
    text_blocks: list[MoeflowTextBlock]

    def to_translate_text(self) -> str:
        result: list[str] = []
        result.append(f">>>>[{self.local_path.name}]<<<<")
        for idx, block in enumerate(self.text_blocks):
            logging.debug(
                "block: %s,%s / %s => %s",
                block.normalized_center_x,
                block.normalized_center_y,
                block.source,
                block.translated,
            )
            position_type = 1
            result.append(
                f"----[{idx}]----[{block.normalized_center_x},{block.normalized_center_y},{position_type}]"
            )
            if block.translated:
                result.append(block.translated)
            elif block.source:
                result.append("(src) " + block.source)
            else:
                result.append("")
        result.append("")
        return "\n".join(result)


class MoeflowProject(BaseModel):
    meta: MoeflowProjectMeta
    files: list[MoeflowFile]

    def to_zip(self, dest: Path):
        translated_texts = "\n".join(
            file.to_translate_text() for file in self.files if file.text_blocks
        )

        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("project.json", self.meta.model_dump_json())
            zf.writestr("translations.txt", translated_texts)
            for file in self.files:
                zf.write(file.local_path, f"images/{file.local_path.name}")
        return dest
