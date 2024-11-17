import manga_translator.textline_merge as textline_merge
import manga_translator.utils.generic as utils_generic
import manga_translator.utils.textblock as utils_textblock
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


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
