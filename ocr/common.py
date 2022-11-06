import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List

from utils import ModelWrapper, Quadrilateral

class CommonOCR(ABC):

    async def recognize(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        '''
        Performs the optical character recognition, using the `textlines` as areas of interests.
        Returns quadrilaterals defined by the `textlines` which contain the recognized text.
        '''
        return await self._recognize(image, textlines, verbose)

    @abstractmethod
    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        pass

class OfflineOCR(CommonOCR, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'ocr')

    async def _recognize(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        pass
