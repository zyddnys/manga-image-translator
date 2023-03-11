from abc import abstractmethod
from typing import List, Tuple
import numpy as np
import cv2
import networkx as nx
import itertools

from .textline_merge import dispatch as dispatch_textline_merge
from ..utils import InfererModule, ModelWrapper, TextBlock, Quadrilateral

class CommonDetector(InfererModule):
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

        text_regions, raw_mask, mask = await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches, verbose)
        text_regions = self._sort_regions(text_regions, image.shape[1], image.shape[0])

        # Remove border
        if new_w > img_w or new_h > img_h:
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = mask[:img_h, :img_w]
                new_text_regions = []
            if raw_mask is not None:
                raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                raw_mask = raw_mask[:img_h, :img_w]
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
        return text_regions, raw_mask, mask

    @abstractmethod
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, det_rearrange_max_batches:int, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass

    def _sort_regions(self, regions: List[TextBlock], width: int, height: int) -> List[TextBlock]:
        # Sort regions from right to left, top to bottom
        sorted_regions = []
        for region in sorted(regions, key=lambda region: region.center[1]):
            for i, sorted_region in enumerate(sorted_regions):
                if region.center[1] > sorted_region.xyxy[3]:
                    continue
                if region.center[1] < sorted_region.xyxy[1]:
                    sorted_regions.insert(i + 1, region)
                    break
                # y center of region inside sorted_region so sort by x instead
                if region.center[0] > sorted_region.center[0]:
                    sorted_regions.insert(i, region)
                    break
            else:
                sorted_regions.append(region)
        return sorted_regions

class OfflineDetector(CommonDetector, ModelWrapper):
    _MODEL_SUB_DIR = 'detection'

    async def _detect(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                       unclip_ratio: float, det_rearrange_max_batches: int, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass
