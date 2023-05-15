import numpy as np
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper

class CommonColorizer(InfererModule):
    _VALID_UPSCALE_RATIOS = None

    async def colorize(self, image: np.ndarray) -> np.ndarray:
        return await self._colorize(image)

    @abstractmethod
    async def _colorize(self, image: np.ndarray) -> np.ndarray:
        pass

class OfflineColorizer(CommonColorizer, ModelWrapper):
    _MODEL_SUB_DIR = 'colorization'

    async def _colorize(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: np.ndarray) -> np.ndarray:
        pass
