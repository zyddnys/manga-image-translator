from pathlib import Path
from ._ocr_google_gemini_multimodal import ImageProcessResult
from moeflow_companion.utils import read_image_dim
import datetime
from moeflow_companion.data import (
    MoeflowProjectMeta,
    MoeflowTextBlock,
    MoeflowProject,
    MoeflowFile,
)


def export_moeflow_project(
    image_files: list[Path],
    process_result: list[ImageProcessResult],
    project_name: str | None,
    dest_dir: Path,
) -> Path:
    if not project_name:
        project_name = image_files[0].name.rsplit(".", 1)[0]
    meta = MoeflowProjectMeta(
        name=project_name,
        intro=f"processed by moeflow companion {datetime.datetime.now().isoformat()}",
    )
    proj = MoeflowProject(
        meta=meta,
        files=_convert_files(image_files, process_result),
    )
    dest = dest_dir / f"{project_name}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    return proj.to_zip(dest)


def _convert_files(
    image_files: list[Path],
    process_result: list[ImageProcessResult],
) -> MoeflowProject:
    files = []
    for image_file, result in zip(image_files, process_result):
        image_w, image_h = read_image_dim(image_file)

        files.append(
            MoeflowFile(
                local_path=image_file,
                image_w=image_w,
                image_h=image_h,
                text_blocks=[
                    MoeflowTextBlock(
                        source=block.text,
                        translated=block.translation,
                        center_x=block.left + block.width / 2.0,
                        center_y=block.top + block.height / 2.0,
                    )
                    for block in result.items
                ],
            )
        )
    return files
