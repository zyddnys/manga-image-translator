"""_summary_"""

from pathlib import Path
from pydantic import BaseModel, Field
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
    center_x: float | int
    center_y: float | int
    source: str | None
    translated: str | None


class MoeflowFile(BaseModel):
    local_path: Path
    image_w: int
    image_h: int
    text_blocks: list[MoeflowTextBlock]

    def to_translate_text(self) -> str:
        result: list[str] = []
        result.append(f">>>>[{self.local_path.name}]<<<<")
        for idx, block in enumerate(self.text_blocks):
            x = float(block.center_x) / self.image_h
            y = float(block.center_y) / self.image_w
            logging.debug(
                "block: %s,%s / %s",
                block.center_x,
                block.center_y,
                block.source,
                block.translated,
            )
            position_type = 1
            result.append(f"----[{idx}]----[{x},{y},{position_type}]")
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
