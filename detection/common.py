from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import cv2
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
        # Add border to small-sized images (instead of simply resizing) due to them likely containing large fonts
        bordered_image_size = detect_size // 2
        img_h, img_w = image.shape[:2]
        new_w, new_h = img_w, img_h
        if min(img_w, img_h) < bordered_image_size:
            new_w = new_h = max(img_w, img_h, bordered_image_size)
            new_image = np.zeros([new_h, new_w, 3]).astype(np.uint8)
            # new_image[:] = np.array([255, 255, 255], np.uint8)
            new_image[:img_h, :img_w] = image
            image = new_image

        text_regions, mask = await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches, verbose)

        # Remove border
        if new_w > img_w or new_h > img_h:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = mask[:img_h, :img_w]
            new_text_regions = []
            # Filter out regions within the border and clamp the points of the remaining regions
            for region in text_regions:
                if region.xyxy[0] >= img_w and region.xyxy[1] >= img_h:
                    continue
                region.xyxy[2] = min(region.xyxy[2], img_w)
                region.xyxy[3] = min(region.xyxy[3], img_h)
                for line in region.lines:
                    for pt in line:
                        pt[0] = min(pt[0], img_w)
                        pt[1] = min(pt[1], img_h)
                new_text_regions.append(region)
            text_regions = new_text_regions
        return text_regions, mask


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
