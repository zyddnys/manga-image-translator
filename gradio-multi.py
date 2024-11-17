import logging
from typing import List
import gradio as gr
import asyncio
from pathlib import Path
import json
import uuid
from PIL import Image
import manga_translator.detection as mit_detection
import manga_translator.ocr as mit_ocr
import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
from manga_translator.gradio import (
    mit_detect_text_default_params,
    mit_ocr_default_params,
    storage_dir,
    MitJSONEncoder,
)
from manga_translator.utils.textblock import TextBlock

STORAGE_DIR_RESOLVED = storage_dir.resolve()

if gr.NO_RELOAD:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for name in ["httpx"]:
        logging.getLogger(name).setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def copy_files(gradio_temp_files: list[str]) -> list[str]:
    new_root: Path = storage_dir / uuid.uuid4().hex
    new_root.mkdir(parents=True, exist_ok=True)

    ret: list[str] = []
    for f in gradio_temp_files:
        new_file = new_root / f.split("/")[-1]
        new_file.write_bytes(Path(f).read_bytes())
        ret.append(str(new_file.relative_to(storage_dir)))
        logger.debug("copied %s to %s", f, new_file)

    return ret


def log_file(basename: str, result: List[TextBlock]):
    logger.debug("file: %s", basename)
    for i, b in enumerate(result):
        logger.debug("  block %d: %s", i, b.text)


async def process_files(
    filename_list: list[str], detector_key: str, ocr_key: str, device: str
) -> str:
    path_list: list[Path] = []
    for f in filename_list:
        assert f
        # p = (storage_dir / f).resolve()
        # assert p.is_file() and STORAGE_DIR_RESOLVED in p.parents, f"illegal path: {f}"
        path_list.append(Path(f))

    await mit_detection.prepare(detector_key)
    await mit_ocr.prepare(ocr_key, device)

    result = await asyncio.gather(
        *[process_file(p, detector_key, ocr_key, device) for p in path_list]
    )

    for r in result:
        log_file(r["filename"], r["text_blocks"])

    return json.dumps(result, cls=MitJSONEncoder)


async def process_file(
    img_path: Path, detector: str, ocr_key: str, device: str
) -> dict:
    pil_img = Image.open(img_path)
    img, mask = utils_generic.load_image(pil_img)
    img_w, img_h = img.shape[:2]

    try:
        # detector
        detector_args = {
            **mit_detect_text_default_params,
            "detector_key": detector,
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
        text_blocks = []
    else:
        logger.debug("processed %s", img_path)

    return {
        "filename": img_path.name,
        "text_blocks": text_blocks,
    }


with gr.Blocks() as demo:
    demo.enable_queue = True
    file_input = gr.File(label="upload file", file_count="multiple", type="filepath")

    ocr_output = gr.JSON(
        label="OCR output",
    )

    device_input = gr.Radio(choices=["cpu", "cuda"], label="device", value="cuda")
    detector_key_input = gr.Radio(
        choices=[
            "default",
            # maybe broken: manga_translator.utils.inference.InvalidModelMappingException: [DBConvNextDetector->model] Invalid _MODEL_MAPPING - Malformed url property
            #  "dbconvnext",
            "ctd",
            "craft",
            "none",
        ],
        value="default",
        label="detector",
    )

    ocr_key_input = gr.Radio(
        choices=["48px", "48px_ctc", "mocr"], label="ocr", value="48px"
    )
    run_button = gr.Button("upload + text detection + OCR + textline_merge")

    @run_button.click(
        inputs=[file_input, detector_key_input, ocr_key_input, device_input],
        outputs=[ocr_output],
    )
    async def on_run_button(
        gradio_temp_files: list[str], detector_key: str, ocr_key: str, device: str
    ) -> str:
        res = await process_files(gradio_temp_files, detector_key, ocr_key, device)
        return res


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
    )
