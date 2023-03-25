from abc import abstractmethod
from typing import List, Tuple
from collections import Counter
import numpy as np
import cv2

from .textline_merge import dispatch as dispatch_textline_merge
from ..utils import InfererModule, ModelWrapper, TextBlock, Quadrilateral


class CommonDetector(InfererModule):
    async def _merge_textlines(self, textlines: List[Quadrilateral], img_width: int, img_height: int, verbose: bool = False) -> List[TextBlock]:
        return await dispatch_textline_merge(textlines, img_width, img_height, verbose)

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                     invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        '''
        Returns textblock list and text mask.
        '''

        # Apply filters
        img_h, img_w = image.shape[:2]
        orig_image = image.copy()
        minimum_image_size = 400
        add_border = min(img_w, img_h) < minimum_image_size
        # Add border if image too small (instead of simply resizing due to them likely containing large fonts)
        if rotate:
            self.logger.debug('Adding rotation')
            image = self._add_rotation(image)
        if add_border:
            self.logger.debug('Adding border')
            image = self._add_border(image, minimum_image_size)
        if invert:
            self.logger.debug('Adding invertion')
            image = self._add_invertion(image)
        if gamma_correct:
            self.logger.debug('Adding gamma correction')
            image = self._add_gamma_correction(image)
        # if True:
        #     self.logger.debug('Adding histogram equalization')
        #     image = self._add_histogram_equalization(image)

        # cv2.imwrite('histogram.png', image)
        # cv2.waitKey(0)

        # Run detection
        text_regions, raw_mask, mask = await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose)
        text_regions = self._sort_regions(text_regions, image.shape[1], image.shape[0])

        # Remove filters
        if add_border:
            text_regions, raw_mask, mask = self._remove_border(image, img_w, img_h, text_regions, raw_mask, mask)
        if auto_rotate:
            # Rotate if horizontal aspect ratios are prevalent to optentially improve detection
            if len(text_regions) > 0:
                orientations = ['h' if region.polygon_aspect_ratio > 75 else 'v' for region in text_regions]
                majority_orientation = Counter(orientations).most_common(1)[0][0]
            else:
                majority_orientation = 'h'
            if majority_orientation == 'h':
                self.logger.info('Rerunning detection with 90° rotation')
                return await self.detect(orig_image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct, rotate=(not rotate), auto_rotate=False, verbose=verbose)
        if rotate:
            text_regions, raw_mask, mask = self._remove_rotation(text_regions, raw_mask, mask)

        return text_regions, raw_mask, mask

    @abstractmethod
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass

    def _add_border(self, image: np.ndarray, target_side_length: int):
        old_h, old_w = image.shape[:2]
        new_w = new_h = max(old_w, old_h, target_side_length)
        new_image = np.zeros([new_h, new_w, 3]).astype(np.uint8)
        # new_image[:] = np.array([255, 255, 255], np.uint8)
        x, y = 0, 0
        # x, y = (new_h - old_h) // 2, (new_w - old_w) // 2
        new_image[y:y+old_h, x:x+old_w] = image
        return new_image

    def _remove_border(self, image: np.ndarray, old_w: int, old_h: int, text_regions, raw_mask, mask):
        new_h, new_w = image.shape[:2]
        raw_mask = cv2.resize(raw_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        raw_mask = raw_mask[:old_h, :old_w]
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = mask[:old_h, :old_w]

        # Filter out regions within the border and clamp the points of the remaining regions
        new_text_regions = []
        for region in text_regions:
            if region.xyxy[0] >= old_w and region.xyxy[1] >= old_h:
                continue
            region.xyxy[2] = min(region.xyxy[2], old_w)
            region.xyxy[3] = min(region.xyxy[3], old_h)
            for line in region.lines:
                for pt in line:
                    pt[0] = min(pt[0], old_w)
                    pt[1] = min(pt[1], old_h)
            new_text_regions.append(region)
        return new_text_regions, raw_mask, mask

    def _add_rotation(self, image: np.ndarray):
        return np.rot90(image, k=-1)

    def _remove_rotation(self, text_regions, raw_mask, mask):
        raw_mask = np.ascontiguousarray(np.rot90(raw_mask))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask).astype(np.uint8))

        for i, region in enumerate(text_regions):
            rot_lines = region.lines[:,:,[1,0]]
            rot_lines[:,:,1] = -rot_lines[:,:,1] + raw_mask.shape[0]
            # TODO: Copy over all values
            new_region = TextBlock(rot_lines, font_size=region.font_size, angle=region.angle, prob=region.prob,
                                   fg_color=region.fg_colors, bg_color=region.bg_colors)
            text_regions[i] = new_region
        return text_regions, raw_mask, mask

    def _add_invertion(self, image: np.ndarray):
        return cv2.bitwise_not(image)

    def _add_gamma_correction(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid * 255) / np.log(mean)
        img_gamma = np.power(image, gamma).clip(0,255).astype(np.uint8)
        return img_gamma

    def _add_histogram_equalization(self, image: np.ndarray):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

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
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                       unclip_ratio: float, verbose: bool = False) -> Tuple[List[TextBlock], np.ndarray]:
        pass
