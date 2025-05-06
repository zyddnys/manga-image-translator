import itertools
import logging
from pathlib import Path
import datetime
from moeflow_companion.data import (
    MoeflowProjectMeta,
    MoeflowTextBlock,
    MoeflowProject,
    MoeflowFile,
)
from ._model import FileBatchProcessResult, FileProcessResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def export_moeflow_project(
    process_result: FileBatchProcessResult, project_name: str | None, dest_dir: Path
) -> Path:
    """_summary_

    Args:
        process_result (FileBatchProcessResult): process result of mit workflow
        project_name (str | None): _description_
        dest (Path): path to the zip file to be created

    Returns:
        Path: _description_
    """
    if not project_name:
        project_name = process_result.files[0].local_path.name.rsplit(".", 1)[0]

    meta = MoeflowProjectMeta(
        name=project_name,
        intro=f"processed by moeflow companion {datetime.datetime.now().isoformat()}",
    )

    proj = MoeflowProject(
        meta=meta,
        files=list(map(_convert_file, process_result.files)),
    )

    dest = dest_dir / f"{project_name}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    return proj.to_zip(dest)


def _convert_file(f: FileProcessResult) -> MoeflowFile:
    if f.translated:
        # there should be only 1 target language
        translated_texts: list[str] = next(iter(f.translated.values()))
    else:
        translated_texts = []
    return MoeflowFile(
        local_path=f.local_path,
        image_w=f.image_w,
        image_h=f.image_h,
        text_blocks=[
            MoeflowTextBlock(
                center_x=block.center_x,
                center_y=block.center_y,
                normalized_center_x=block.center_x / f.image_w,
                normalized_center_y=block.center_y / f.image_h,
                source=block.text,
                translated=translated,
            )
            for block, translated in itertools.zip_longest(
                f.text_blocks, translated_texts
            )
        ],
    )
