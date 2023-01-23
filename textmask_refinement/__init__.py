from typing import List
import cv2
import numpy as np

from utils import Quadrilateral
from detection.ctd_utils import TextBlock
from .text_mask_utils import complete_mask_fill, filter_masks, complete_mask

async def dispatch(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', verbose: bool = False) -> np.ndarray:
    img_resized = cv2.resize(raw_image, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (raw_image.shape[1] // 2, raw_image.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
    mask_resized[mask_resized > 0] = 255
    bboxes_resized = []
    for region in text_regions:
        for l in region.lines:
            a = Quadrilateral(l, '', 0)
            bboxes_resized.append((a.aabb.x // 2, a.aabb.y // 2, a.aabb.w // 2, a.aabb.h // 2))
    mask_ccs, cc2textline_assignment = filter_masks(mask_resized, bboxes_resized)
    if mask_ccs:
        #mask_filtered = reduce(cv2.bitwise_or, mask_ccs)
        #cv2.imwrite(f'result/{task_id}/mask_filtered.png', mask_filtered)
        #cv2.imwrite(f'result/{task_id}/mask_filtered_img.png', overlay_mask(img_resized_2, mask_filtered))
        if method == 'fit_text':
            final_mask = complete_mask(img_resized, mask_ccs, bboxes_resized, cc2textline_assignment)
        else:
            final_mask = complete_mask_fill(img_resized, mask_ccs, bboxes_resized, cc2textline_assignment)
        #cv2.imwrite(f'result/{task_id}/mask.png', final_mask)
        #cv2.imwrite(f'result/{task_id}/mask_img.png', overlay_mask(img_resized_2, final_mask))
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255
    else:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    return final_mask
