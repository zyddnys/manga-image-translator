import gradio as gr
import numpy as np


import dotenv
import logging
import asyncio
import manga_translator.detection as detection
import manga_translator.ocr as ocr
import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock
from manga_translator.gradio import DetectionState, mit_detect_text_default_params
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if gr.NO_RELOAD:
    logging.basicConfig(level=logging.INFO, force=True)
    for name in ["httpx"]:
        logging.getLogger(name).setLevel(logging.WARN)

dotenv.load_dotenv()


mit_ocr_default_params = dict(
    ocr_key="48px",  # recommended by rowland
    # ocr_key="48px_ctc",
    # ocr_key="mocr",  # XXX: mocr may have different output format
    # use_mocr_merge=True,
    verbose=True,
)


class DetectionResult(TypedDict):
    textlines: List[utils_generic.Quadrilateral]
    mask_raw: np.ndarray
    mask: np.ndarray | None


async def run_detection_single_image(
    image: np.ndarray, detector_key: str
) -> DetectionResult:
    print("image", image.shape)
    textlines, mask_raw, mask = await detection.dispatch(
        image=image, **{"detector_key": detector_key, **mit_detect_text_default_params}
    )
    print("textlines", textlines)
    print("mask_raw", mask_raw)
    print("mask", mask)
    return {
        "textlines": textlines,
        "mask_raw": mask_raw,
        "mask": mask,
    }


input_single_img = gr.Image(label="input image")
output_json = gr.JSON(label="output json")

with gr.Blocks() as demo:
    gr.Markdown(
        """
# manga-image-translator demo
                """.strip()
    )

    detector_state = gr.State(DetectionState())

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Detection")
            img_file = gr.Image(label="input image", height=256, width=256)
            detector_key = gr.Radio(
                choices=["default", "dbconvnext", "ctd", "craft", "none"],
                label="detector key",
            )

            btn_detect = gr.Button("detect")
            detector_state_dump = gr.TextArea(
                label="detection state"  # , value=lambda: repr(detector_state.value)
            )
        with gr.Column():
            gr.Markdown("## OCR")
            ocr_key = gr.Radio(choices=["48px", "48px_ctc", "mocr"], label="ocr key")
            btn_ocr = gr.Button("ocr")
            ocr_state_dump = gr.TextArea(label="ocr state")

    @btn_detect.click(
        inputs=[detector_state, img_file, detector_key],
        outputs=[detector_state, detector_state_dump],
    )
    async def run_detector(
        prev: DetectionState | gr.State,
        img,
        detector_key: Optional[str],
    ):
        # print("prev", prev)
        prev_value = prev if isinstance(prev, DetectionState) else prev.value
        logger.debug("run_detector %s %s", prev_value, type(img))

        value = prev_value.copy(img=img)

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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
