from manga_translator.gradio.detection import DetectionState
from manga_translator.gradio.ocr import OcrState


def create_json(detection_state: DetectionState, ocr_state: OcrState) -> dict:
    return {
        "img": img_bytes,
        "detection_state": detection_state.__json__(),
        "ocr_state": ocr_state.__json__(),
    }
