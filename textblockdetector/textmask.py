from os import stat
from typing import List
import cv2
import numpy as np
from .textblock import TextBlock
from .utils.imgproc_utils import draw_connected_labels, expand_textwindow, union_area

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LANG_ENG = 0
LANG_JPN = 1

REFINEMASK_INPAINT = 0
REFINEMASK_ANNOTATION = 1

def get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or bin < bin_tol:
                break
    return top_colors

def minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    e_size = 1
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    if neg_xor_sum < xor_sum:
        return neg_threshed, neg_xor_sum
    else:
        return threshed, xor_sum

def get_otsuthresh_masklist(img, pred_mask, per_channel=False) -> List[np.ndarray]:
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for c in channels:
        _, threshed = cv2.threshold(c, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        threshed, xor_sum = minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda x: x[1])
    if per_channel:
        return mask_list
    else:
        return [mask_list[0]]

def get_topk_masklist(im_grey, pred_mask):
    if len(im_grey.shape) == 3 and im_grey.shape[-1] == 3:
        im_grey = cv2.cvtColor(im_grey, cv2.COLOR_BGR2GRAY)
    msk = np.ascontiguousarray(pred_mask)
    candidate_grey_px = im_grey[np.where(cv2.erode(msk, np.ones((3,3), np.uint8), iterations=1) > 127)]
    bin, his = np.histogram(candidate_grey_px, bins=255)
    topk_color = get_topk_color(his, bin, color_var=10, k=3)
    color_range = 30
    mask_list = list()
    for ii, color in enumerate(topk_color):
        c_top = min(color+color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(im_grey, c_bottom, c_top)
        threshed, xor_sum = minxor_thresh(threshed, msk)
        mask_list.append([threshed, xor_sum])
    return mask_list

def merge_mask_list(mask_list, pred_mask, blk: TextBlock = None, pred_thresh=30, text_window=None, filter_with_lines=False, refine_mode=REFINEMASK_INPAINT):
    mask_list.sort(key=lambda x: x[1])
    linemask = None
    if blk is not None and filter_with_lines:
        linemask = np.zeros_like(pred_mask)
        lines = blk.lines_array(dtype=np.int64)
        for line in lines:
            line[..., 0] -= text_window[0]
            line[..., 1] -= text_window[1]
            cv2.fillPoly(linemask, [line], 255)
        linemask = cv2.dilate(linemask, np.ones((3, 3), np.uint8), iterations=3)
    
    if pred_thresh > 0:
        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))      
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)
    connectivity = 8
    mask_merged = np.zeros_like(pred_mask)
    for ii, (candidate_mask, xor_sum) in enumerate(mask_list):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)
        for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
            if label_index != 0: # skip background label
                x, y, w, h, area = stat
                if w * h < 3:
                    continue
                x1, y1, x2, y2 = x, y, x+w, y+h
                label_local = labels[y1: y2, x1: x2]
                label_cordinates = np.where(label_local==label_index)
                tmp_merged = np.zeros_like(label_local, np.uint8)
                tmp_merged[label_cordinates] = 255
                tmp_merged = cv2.bitwise_or(mask_merged[y1: y2, x1: x2], tmp_merged)
                xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1: y2, x1: x2]).sum()
                xor_origin = cv2.bitwise_xor(mask_merged[y1: y2, x1: x2], pred_mask[y1: y2, x1: x2]).sum()
                if xor_merged < xor_origin:
                    mask_merged[y1: y2, x1: x2] = tmp_merged

    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((5, 5), np.uint8), iterations=1)
    # fill holes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-mask_merged, connectivity, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    if len(sorted_area) > 1:
        area_thresh = sorted_area[-2]
    else:
        area_thresh = sorted_area[-1]
    for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
        x, y, w, h, area = stat
        if area < area_thresh:
            x1, y1, x2, y2 = x, y, x+w, y+h
            label_local = labels[y1: y2, x1: x2]
            label_cordinates = np.where(label_local==label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[label_cordinates] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1: y2, x1: x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1: y2, x1: x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1: y2, x1: x2], pred_mask[y1: y2, x1: x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1: y2, x1: x2] = tmp_merged
    return mask_merged


def refine_undetected_mask(img: np.ndarray, mask_pred: np.ndarray, mask_refined: np.ndarray, blk_list: List[TextBlock], refine_mode=REFINEMASK_INPAINT):
    mask_pred[np.where(mask_refined > 30)] = 0
    _, pred_mask_t = cv2.threshold(mask_pred, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list = []
    if len(valid_labels) > 0:
        for lab_index in valid_labels[1:]:
            x, y, w, h, area = stats[lab_index]
            bx1, by1 = x, y
            bx2, by2 = x+w, y+h
            bbox = [bx1, by1, bx2, by2]
            bbox_score = -1
            for blk in blk_list:
                bbox_s = union_area(blk.xyxy, bbox)
                if bbox_s > bbox_score:
                    bbox_score = bbox_s
            if bbox_score / w / h < 0.5:
                seg_blk_list.append(TextBlock(bbox))
    if len(seg_blk_list) > 0:
        mask_refined = cv2.bitwise_or(mask_refined, refine_mask(img, mask_pred, seg_blk_list, refine_mode=refine_mode))
    return mask_refined


def refine_mask(img: np.ndarray, pred_mask: np.ndarray, blk_list: List[TextBlock], refine_mode: int = REFINEMASK_INPAINT) -> np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    for blk in blk_list:
        bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy, expand_r=16)
        im = np.ascontiguousarray(img[by1: by2, bx1: bx2])
        msk = np.ascontiguousarray(pred_mask[by1: by2, bx1: bx2])
        mask_list = get_topk_masklist(im, msk)
        mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
        mask_merged = merge_mask_list(mask_list, msk, blk=blk, text_window=[bx1, by1, bx2, by2], refine_mode=refine_mode)
        mask_refined[by1: by2, bx1: bx2] = cv2.bitwise_or(mask_refined[by1: by2, bx1: bx2], mask_merged)
    return mask_refined

