import gradio as gr
import numpy as np
from PIL import Image


import dotenv
import logging
import os.path
from pathlib import Path
import manga_translator.detection as detection
import manga_translator.ocr as mit_ocr
import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
from manga_translator.gradio import (
    DetectionState,
    OcrState,
    mit_detect_text_default_params,
)
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if gr.NO_RELOAD:
    logging.basicConfig(level=logging.INFO, force=True)
    for name in ["httpx"]:
        logging.getLogger(name).setLevel(logging.WARN)

dotenv.load_dotenv()


with gr.Blocks() as demo:
    gr.Markdown(
        """
# manga-image-translator demo
                """.strip()
    )

    detector_state = gr.State(DetectionState())
    ocr_state = gr.State(OcrState())

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Detection")
            img_file = gr.Image(
                label="input image", height=256, width=256, type="filepath"
            )
            detector_key = gr.Radio(
                choices=[
                    "default",
                    # maybe broken: manga_translator.utils.inference.InvalidModelMappingException: [DBConvNextDetector->model] Invalid _MODEL_MAPPING - Malformed url property
                    #  "dbconvnext",
                    "ctd",
                    "craft",
                    "none",
                ],
                label="detector",
            )

            btn_detect = gr.Button("run detector")
            detector_state_dump = gr.TextArea(
                label="detector result"  # , value=lambda: repr(detector_state.value)
            )
        with gr.Column():
            gr.Markdown("## OCR")
            ocr_key = gr.Radio(choices=["48px", "48px_ctc", "mocr"], label="ocr")
            btn_ocr = gr.Button("ocr")
            ocr_state_dump = gr.TextArea(label="ocr state")

    @btn_detect.click(
        inputs=[detector_state, img_file, detector_key],
        outputs=[detector_state, detector_state_dump],
    )
    async def run_detector(
        prev: DetectionState | gr.State,
        img_path: Optional[str],
        detector_key: Optional[str],
    ):
        # print("prev", prev)
        prev_value = prev if isinstance(prev, DetectionState) else None  # prev.value
        assert prev_value, "prev_value is None"
        logger.debug("run_detector %s %s", prev_value, img_path)

        value = prev_value.copy()

        if img_path:
            raw_bytes = Path(img_path).read_bytes()
            pil_img = Image.open(img_path)
            img, mask = utils_generic.load_image(pil_img)
            value = value.copy(
                raw_filename=os.path.basename(img_path), raw_bytes=raw_bytes, img=img
            )
        else:
            value = prev_value.copy(raw_filename=None, raw_bytes=None, img=None)

        if detector_key:
            value = value.copy(
                args={**mit_detect_text_default_params, "detector_key": detector_key}
            )

        if value.img is not None and value.args is not None:
            logger.debug("run inference")
            textlines, mask_raw, mask = await detection.dispatch(
                image=img, **value.args
            )
            value = value.copy(textlines=textlines, mask_raw=mask_raw, mask=mask)

        logger.debug("run_detector result %s", value)
        return value, repr(value)

    @btn_ocr.click(
        inputs=[ocr_state, detector_state, ocr_key],
        outputs=[ocr_state, ocr_state_dump],
    )
    async def run_ocr(
        prev_value: OcrState,
        detector_state: DetectionState,
        ocr_key: Optional[str],
    ):
        logger.debug(
            "run ocr %s %s %s", type(prev_value), type(detector_state), ocr_key
        )

        if not (
            ocr_key and (detector_state.img is not None) and detector_state.textlines
        ):
            return prev_value, repr(prev_value)

        textlines = await mit_ocr.dispatch(
            ocr_key=ocr_key,
            image=detector_state.img,
            regions=detector_state.textlines,
            args={},
            verbose=True,
        )

        img_w, img_h = detector_state.img.shape[:2]
        text_blocks = await textline_merge.dispatch(
            textlines=textlines, width=img_w, height=img_h
        )

        value = prev_value.copy(text_blocks=text_blocks, ocr_key=ocr_key)
        return value, repr(value)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
