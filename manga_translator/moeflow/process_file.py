import asyncio
import uuid
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
from .model import FileProcessResult, TextBlock
from .translate import translate_text
from threading import Lock
import itertools

load_model_mutex = Lock()

_storage_dir = Path(__file__).parent.parent.parent / "storage"


logger = logging.getLogger(__name__)

storage_dir = _storage_dir.resolve()

logger.info("temp storage dir: %s", storage_dir)


def log_file(f: FileProcessResult):
    logger.info("file: %s", f.local_path.name)
    for i, (b, translated) in enumerate(
        itertools.zip_longest(f.text_blocks, f.translated or [])
    ):
        logger.info("  block %d: %s => %s", i, b.text, translated)


def copy_files(gradio_temp_files: list[str]) -> list[Path]:
    new_root: Path = _storage_dir / uuid.uuid4().hex
    new_root.mkdir(parents=True, exist_ok=True)

    ret: list[str] = []
    for f in gradio_temp_files:
        new_file = new_root / f.split("/")[-1]
        new_file.write_bytes(Path(f).read_bytes())
        ret.append(new_file.relative_to(_storage_dir))
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
    pil_img = Image.open(storage_dir / img_path)
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
        ocr_key=ocr_key,
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
) -> list[FileProcessResult]:
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
                translator_key="gpt4",
                target_language=target_language,
            )
            for p in path_list
        ]
    )

    for r in results:
        log_file(r)

    return results
