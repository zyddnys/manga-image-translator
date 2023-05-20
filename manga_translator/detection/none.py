import numpy as np

from .common import CommonDetector

class NoneDetector(CommonDetector):
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float, verbose: bool = False):
        return [], np.zeros(image.shape), None
