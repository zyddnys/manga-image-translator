from pydantic import BaseModel
from .model import FileBatchProcessResult, FileProcessResult
from ._const import create_unique_dir
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def export_moeflow_project(
    process_result: FileBatchProcessResult, project_name: str | None
) -> Path:
    if not project_name:
        project_name = process_result.files[0].local_path.name.rsplit(".", 1)[0]

    meta_json = MoeflowProjectMeta(name=project_name, intro="").model_dump_json()

    output_dir = create_unique_dir("moeflow-export")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{project_name}.zip"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_file, "w") as zf:
        zf.writestr("project.json", meta_json)
        zf.writestr("translations.txt", _build_file(process_result.files))
        for f in process_result.files:
            zf.write(f.local_path, f"images/{f.local_path.name}")
    return output_file


class MoeflowProjectMeta(BaseModel):
    name: str
    intro: str
    output_language: str = "en"
    default_role: str = "supporter"
    allow_apply_type: int = 3
    application_check_type: int = 1
    is_need_check_application: bool = False
    source_language: str = "ja"


def _build_file(files: list[FileProcessResult]) -> str:
    result: list[str] = []
    for file in files:
        result.append(f">>>>[{file.local_path.name}]<<<<")
        if file.translated:
            translated_texts: list[str] = next(iter(file.translated.values()))
        else:
            translated_texts = None
        logging.debug(
            "file: %s %s x %s", file.local_path.name, file.image_w, file.image_h
        )

        for idx, block in enumerate(file.text_blocks):
            t = translated_texts[idx] if translated_texts else ""
            x = (
                block.center_x / file.image_h
            )  # this works but IDK why. Is mit using a swapped coordinate system?
            y = block.center_y / file.image_w
            logging.debug(
                "block: %s,%s / %s", block.center_x, block.center_y, block.text
            )
            position_type = 1
            result.append(f"----[{idx}]----[{x},{y},{position_type}]")
            logging.debug("serialized block: %s,%s / %s", x, y, result[-1])
            result.append(t)
    return "\n".join(result + [""])
