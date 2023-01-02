from abc import ABC, abstractmethod
from typing import List
import numpy as np
import os

from utils import ModelWrapper, Quadrilateral
from .textline_merge import dispatch as dispatch_textline_merge
from .textmask_refinement import dispatch as dispatch_mask_refinement
from .ctd_utils import TextBlock

class CommonDetector(ABC):

    async def _merge_textlines(self, textlines: List[Quadrilateral], img_width: int, img_height: int) -> List[TextBlock]:
        return await dispatch_textline_merge(textlines, img_width, img_height)

    async def _refine_textmask(self, textlines: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text') -> np.ndarray:
        return await dispatch_mask_refinement(textlines, raw_image, raw_mask, method)

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, det_rearrange_max_batches: int, verbose: bool = False) -> tuple[list[TextBlock], np.ndarray]:
        '''
        Returns textblock list and text mask.
        '''
        return await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches, verbose)

    @abstractmethod
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, det_rearrange_max_batches:int, verbose: bool = False) -> tuple[list[TextBlock], np.ndarray]:
        pass

class OfflineDetector(CommonDetector, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'detection')

    async def _detect(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                       unclip_ratio: float, det_rearrange_max_batches: int, verbose: bool = False) -> tuple[list[TextBlock], np.ndarray]:
        pass
