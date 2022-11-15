import os
import numpy as np
from abc import ABC, abstractmethod

from utils import ModelWrapper

class CommonInpainter(ABC):

    async def inpaint(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        return await self._inpaint(image, mask, inpainting_size, verbose)

    @abstractmethod
    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass

class OfflineInpainter(CommonInpainter, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'inpainting')

    async def _inpaint(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass
