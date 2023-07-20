from PIL import Image
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper

class CommonColorizer(InfererModule):
    _VALID_UPSCALE_RATIOS = None

    async def colorize(self, image: Image.Image, colorization_size: int, **kwargs) -> Image.Image:
        return await self._colorize(image, colorization_size, **kwargs)

    @abstractmethod
    async def _colorize(self, image: Image.Image, colorization_size: int, **kwargs) -> Image.Image:
        pass

class OfflineColorizer(CommonColorizer, ModelWrapper):
    _MODEL_SUB_DIR = 'colorization'

    async def _colorize(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: Image.Image, colorization_size: int, **kwargs) -> Image.Image:
        pass
