import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock
from dataclasses import dataclass
from typing import List, Optional
from .json_encoder import JSONEncoder

import numpy as np
import json


mit_detect_text_default_params = dict(
    detector_key="default",
    # mostly defaults from manga-image-translator/args.py
    detect_size=2560,
    text_threshold=0.5,
    box_threshold=0.7,
    unclip_ratio=2.3,
    invert=False,
    device="cuda",
    gamma_correct=False,
    rotate=False,
    verbose=True,
)


@dataclass(frozen=True, kw_only=True)
class DetectionState:
    img: Optional[np.ndarray] = None
    args: Optional[dict] = None
    textlines: Optional[List[utils_generic.Quadrilateral]] = None
    mask: Optional[np.ndarray] = None
    mask_raw: Optional[np.ndarray] = None

    def copy(self, **kwargs):
        return DetectionState(
            img=kwargs.get("img", self.img),
            args=kwargs.get("args", self.args),
            textlines=kwargs.get("textlines", self.textlines),
            mask=kwargs.get("mask", self.mask),
            mask_raw=kwargs.get("mask_raw", self.mask_raw),
        )

    def __repr__(self):
        return f"DetectionState(img={type(self.img)}, args={self.args}, textlines={type(self.textlines)}, mask={type(self.mask)}, mask_raw={type(self.mask_raw)})"

    def __json__(self):
        return {
            "img": to_json(self.img),
            "args": to_json(self.args),
            "textlines": to_json(self.textlines),
            "mask": to_json(self.mask),
            "mask_raw": to_json(self.mask_raw),
        }


mit_ocr_default_params = dict(
    ocr_key="48px",  # recommended by rowland
    # ocr_key="48px_ctc",
    # ocr_key="mocr",  # XXX: mocr may have different output format
    # use_mocr_merge=True,
    verbose=True,
)


@dataclass(frozen=True, kw_only=True)
class OcrState:
    text_blocks: Optional[List[utils_textblock.TextBlock]] = None
    ocr_key: Optional[str] = None

    def copy(self, **kwargs):
        return OcrState(
            ocr_key=kwargs.get("ocr_key", self.ocr_key),
            text_blocks=kwargs.get("text_blocks", self.text_blocks),
        )


def to_json(obj):
    return json.loads(json.dumps(obj, cls=JSONEncoder))
