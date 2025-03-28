import asyncio
import logging
from pathlib import Path
from PIL import Image
from .detection import (
    mit_detect_text_default_params,
)

from .ocr import (
    mit_ocr_default_params,
)
import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
import manga_translator.detection as mit_detection
import manga_translator.ocr as mit_ocr
from mit_moeflow.model import FileBatchProcessResult, FileProcessResult, TextBlock
from .translate import translate_text
from ._const import create_unique_dir
from threading import Lock
import itertools

load_model_mutex = Lock()


logger = logging.getLogger(__name__)


def log_file(f: FileProcessResult):
    logger.info("file: %s", f.local_path.name)
    for i, (b, translated) in enumerate(
        itertools.zip_longest(f.text_blocks, f.translated or [])
    ):
        logger.info("  block %d: %s => %s", i, b.text, translated)


def copy_files(gradio_temp_files: list[str]) -> list[Path]:
    new_root: Path = create_unique_dir("upload")
    new_root.mkdir(parents=True, exist_ok=True)

    ret: list[str] = []
    for f in gradio_temp_files:
        new_file = new_root / f.split("/")[-1]
        new_file.write_bytes(Path(f).read_bytes())
        ret.append(new_file)
        logger.debug("copied %s to %s", f, new_file)

    return ret


async def process_file(
    img_path: Path,
    *,
    detector_key: str,
    ocr_key: str,
    device: str,
    translator_key: str | None = None,
    target_language: str | None = None,
) -> FileProcessResult:
    pil_img = Image.open(img_path)
    img, mask = utils_generic.load_image(pil_img)
    img_w, img_h = img.shape[:2]

    try:
        # detector
        detector_args = {
            **mit_detect_text_default_params,
            "detector_key": detector_key,
            "device": device,
        }
        regions, mask_raw, mask = await mit_detection.dispatch(
            image=img, **detector_args
        )
        # ocr
        ocr_args = {**mit_ocr_default_params, "ocr_key": ocr_key, "device": device}
        textlines = await mit_ocr.dispatch(image=img, regions=regions, **ocr_args)
        # textline merge
        text_blocks = await textline_merge.dispatch(
            textlines=textlines, width=img_w, height=img_h
        )
    except Exception as e:
        logger.error("error processing %s: %s", img_path, e)
        print(e)
        text_blocks = []
    else:
        logger.debug("processed %s", img_path)

    text_blocks = [TextBlock.from_mit(t) for t in text_blocks]

    if translator_key is not None and target_language is not None and text_blocks:
        translated = {
            target_language: await translate_text(
                [b.text for b in text_blocks],
                translator_key=translator_key,
                target_lang=target_language,
            )
        }
    else:
        translated = None

    return FileProcessResult(
        local_path=img_path,
        image_w=img_w,
        image_h=img_h,
        detector_key=detector_key,
        text_blocks=text_blocks,
        translator_key=translator_key,
        translated=translated,
    )


async def process_files(
    filename_list: list[str],
    *,
    detector_key: str,
    ocr_key: str,
    device: str,
    # translator_key: str | None = None,
    target_language: str | None = None,
) -> FileBatchProcessResult:
    path_list = copy_files(filename_list)

    with load_model_mutex:
        await mit_detection.prepare(detector_key)
        await mit_ocr.prepare(ocr_key, device)

    results: list[FileProcessResult] = await asyncio.gather(
        *[
            process_file(
                p,
                detector_key=detector_key,
                ocr_key=ocr_key,
                device=device,
                translator_key="chatgpt",
                target_language=target_language,
            )
            for p in path_list
        ]
    )

    for r in results:
        log_file(r)

    return FileBatchProcessResult(
        files=results, target_languages=[target_language] if target_language else None
    )
