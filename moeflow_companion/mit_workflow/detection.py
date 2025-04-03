from dataclasses import dataclass
from typing import List, Optional

from moeflow_companion.data.model import to_json
import manga_translator.utils.generic as utils_generic
import numpy as np

mit_detect_text_default_params = dict(
    detector_key="default",
    # mostly defaults from manga-image-translator/args.py
    detect_size=2560,
    text_threshold=0.5,
    box_threshold=0.7,
    unclip_ratio=2.3,
    invert=False,
    # device="cpu",
    gamma_correct=False,
    rotate=False,
    verbose=True,
)


@dataclass(frozen=True, kw_only=True)
class DetectionState:
    raw_filename: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    img: Optional[np.ndarray] = None
    args: Optional[dict] = None
    textlines: Optional[List[utils_generic.Quadrilateral]] = None
    mask: Optional[np.ndarray] = None
    mask_raw: Optional[np.ndarray] = None

    def copy(self, **kwargs):
        return DetectionState(
            raw_filename=kwargs.get("raw_filename", self.raw_filename),
            raw_bytes=kwargs.get("raw_bytes", self.raw_bytes),
            img=kwargs.get("img", self.img),
            args=kwargs.get("args", self.args),
            textlines=kwargs.get("textlines", self.textlines),
            mask=kwargs.get("mask", self.mask),
            mask_raw=kwargs.get("mask_raw", self.mask_raw),
        )

    def __repr__(self):
        return f"DetectionState(raw_filename={self.raw_filename}, raw_bytes={type(self.raw_bytes)} img={type(self.img)}, args={self.args}, textlines={type(self.textlines)}, mask={type(self.mask)}, mask_raw={type(self.mask_raw)})"

    def __json__(self):
        return {
            "raw_filename": self.raw_filename,
            "raw_bytes": to_json(self.raw_bytes),
            "img": to_json(self.img),
            "args": to_json(self.args),
            "textlines": to_json(self.textlines),
            "mask": to_json(self.mask),
            "mask_raw": to_json(self.mask_raw),
        }
