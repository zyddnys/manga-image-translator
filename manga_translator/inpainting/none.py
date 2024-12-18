import numpy as np

from .common import CommonInpainter
from ..config import InpainterConfig


class NoneInpainter(CommonInpainter):

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        img_inpainted = np.copy(image)
        img_inpainted[mask > 0] = np.array([255, 255, 255], np.uint8)
        return img_inpainted
