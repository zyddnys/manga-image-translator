from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import os

from utils import ModelWrapper, Quadrilateral
from .textline_merge import dispatch as dispatch_textline_merge
from .ctd_utils import TextBlock

class CommonDetector(ABC):

    async def _merge_textlines(self, textlines: List[Quadrilateral], img_width: int, img_height: int, verbose: bool = False) -> List[TextBlock]:
        return await dispatch_textline_merge(textlines, img_width, img_height, verbose)

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, det_rearrange_max_batches: int, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        '''
        Returns textblock list and text mask.
        '''
        return await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches, verbose)

    @abstractmethod
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, det_rearrange_max_batches:int, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass

class OfflineDetector(CommonDetector, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'detection')

    async def _detect(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                       unclip_ratio: float, det_rearrange_max_batches: int, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass
