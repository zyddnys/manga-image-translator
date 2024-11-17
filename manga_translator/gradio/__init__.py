from pathlib import Path
from .ocr import mit_ocr_default_params, OcrState
from .detection import mit_detect_text_default_params, DetectionState
from .json_encoder import JSONEncoder as MitJSONEncoder

storage_dir = Path(__file__).parent.parent / "storage"

__all__ = [
    "mit_ocr_default_params",
    "OcrState",
    "mit_detect_text_default_params",
    "DetectionState",
    "storage_dir",
    "MitJSONEncoder",
]
