from PIL import Image
from typing import List
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper

class CommonUpscaler(InfererModule):
    _VALID_UPSCALE_RATIOS = None

    async def upscale(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        if upscale_ratio == 1:
            return image_batch
        if self._VALID_UPSCALE_RATIOS and upscale_ratio not in self._VALID_UPSCALE_RATIOS:
            new_upscale_ratio = min(self._VALID_UPSCALE_RATIOS, key = lambda x: abs(x - upscale_ratio))
            self.logger.warn(f'Changed upscale ratio {upscale_ratio} to closest supported value: {new_upscale_ratio}')
            upscale_ratio = new_upscale_ratio
        return await self._upscale(image_batch, upscale_ratio)

    @abstractmethod
    async def _upscale(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        pass

class OfflineUpscaler(CommonUpscaler, ModelWrapper):
    _MODEL_SUB_DIR = 'upscaling'

    async def _upscale(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image_batch: List[Image.Image], upscale_ratio: float) -> List[Image.Image]:
        pass
