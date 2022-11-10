from PIL import Image
from typing import List
from abc import ABC, abstractmethod

from utils import ModelWrapper

class CommonUpscaler(ABC):

    async def upscale(self, image_batch: List[Image.Image], upscale_ratio: int) -> List[Image.Image]:
        if upscale_ratio == 1:
            return image_batch
        return await self._upscale(image_batch, upscale_ratio)

    @abstractmethod
    async def _upscale(self, image_batch: List[Image.Image], upscale_ratio: int) -> List[Image.Image]:
        pass

class OfflineUpscaler(CommonUpscaler, ModelWrapper):

    async def _upscale(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image_batch: List[Image.Image], upscale_ratio: int) -> List[Image.Image]:
        pass
