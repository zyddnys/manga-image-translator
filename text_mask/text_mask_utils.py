
from typing import Tuple, List
import numpy as np
import cv2
import math

from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from functools import reduce
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

COLOR_RANGE_SIGMA = 1.5 # how many stddev away is considered the same color

def save_rgb(fn, img) :
	if len(img.shape) == 3 and img.shape[2] == 3 :
		cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	else :
		cv2.imwrite(fn, img)

def area(x1, y1, w1, h1, x2, y2, w2, h2):  # returns None if rectangles don't intersect
	x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
	y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
	return x_overlap * y_overlap

def dist(x1, y1, x2, y2) :
	return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
	left = x2b < x1
	right = x1b < x2
	bottom = y2b < y1
	top = y1b < y2
	if top and left:
		return dist(x1, y1b, x2b, y2)
	elif left and bottom:
		return dist(x1, y1, x2b, y2b)
	elif bottom and right:
		return dist(x1b, y1, x2, y2b)
	elif right and top:
		return dist(x1b, y1b, x2, y2)
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	else:             # rectangles intersect
		return 0

def filter_masks(mask_img: np.ndarray, text_lines: List[Tuple[int, int, int, int]], keep_threshold = 1e-2) :
	mask_img = mask_img.copy()
	for (x, y, w, h) in text_lines :
		cv2.rectangle(mask_img, (x, y), (x + w, y + h), (0), 1)
	if len(text_lines) == 0 :
		return [], []
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img)

	cc2textline_assignment = []
	result = []
	M = len(text_lines)
	ratio_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
	dist_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
	for i in range(1, num_labels) :
		if stats[i, cv2.CC_STAT_AREA] <= 9 :
			continue # skip area too small
		cc = np.zeros_like(mask_img)
		cc[labels == i] = 255
		x1, y1, w1, h1 = cv2.boundingRect(cc)
		area1 = w1 * h1
		for j in range(M) :
			x2, y2, w2, h2 = text_lines[j]
			area2 = w2 * h2
			overlapping_area = area(x1, y1, w1, h1, x2, y2, w2, h2)
			ratio_mat[i, j] = overlapping_area / min(area1, area2)
			dist_mat[i, j] = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
		j = np.argmax(ratio_mat[i])
		unit = min([h1, w1, h2, w2])
		if ratio_mat[i, j] > keep_threshold :
			cc2textline_assignment.append(j)
			result.append(np.copy(cc))
		else :
			j = np.argmin(dist_mat[i])
			if dist_mat[i, j] < 0.5 * unit :
				cc2textline_assignment.append(j)
				result.append(np.copy(cc))
			else :
				# discard
				pass
	return result, cc2textline_assignment

from pydensecrf.utils import compute_unary, unary_from_softmax
import pydensecrf.densecrf as dcrf

def refine_mask(rgbim, rawmask) :
	if len(rawmask.shape) == 2 :
		rawmask = rawmask[:, :, None]
	mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
	mask_softmax = mask_softmax.astype(np.float32) / 255.0
	n_classes = 2
	feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
	unary = unary_from_softmax(feat_first)
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF2D(rgbim.shape[1], rgbim.shape[0], n_classes)

	d.setUnaryEnergy(unary)
	d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NO_NORMALIZATION)

	d.addPairwiseBilateral(sxy=23, srgb=7, rgbim=rgbim,
						compat=20,
						kernel=dcrf.DIAG_KERNEL,
						normalization=dcrf.NO_NORMALIZATION)
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((rgbim.shape[0], rgbim.shape[1]))
	crf_mask = np.array(res * 255, dtype=np.uint8)
	return crf_mask

def complete_mask_fill(img_np: np.ndarray, ccs: List[np.ndarray], text_lines: List[Tuple[int, int, int, int]], cc2textline_assignment) :
	if len(ccs) == 0 :
		return
	for (x, y, w, h) in text_lines :
		final_mask = cv2.rectangle(final_mask, (x, y), (x + w, y + h), (255), -1)
	return final_mask

def complete_mask(img_np: np.ndarray, ccs: List[np.ndarray], text_lines: List[Tuple[int, int, int, int]], cc2textline_assignment) :
	if len(ccs) == 0 :
		return
	textline_ccs = [np.zeros_like(ccs[0]) for _ in range(len(text_lines))]
	for i, cc in enumerate(ccs) :
		txtline = cc2textline_assignment[i]
		textline_ccs[txtline] = cv2.bitwise_or(textline_ccs[txtline], cc)
	final_mask = np.zeros_like(ccs[0])
	img_np = cv2.bilateralFilter(img_np, 17, 80, 80)
	for i, cc in enumerate(tqdm(textline_ccs)) :
		x1, y1, w1, h1 = cv2.boundingRect(cc)
		text_size = min(w1, h1)
		extend_size = int(text_size * 0.1)
		x1 = max(x1 - extend_size, 0)
		y1 = max(y1 - extend_size, 0)
		w1 += extend_size * 2
		h1 += extend_size * 2
		w1 = min(w1, img_np.shape[1] - x1 - 1)
		h1 = min(h1, img_np.shape[0] - y1 - 1)
		dilate_size = max((int(text_size * 0.3) // 2) * 2 + 1, 3)
		kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
		cc_region = np.ascontiguousarray(cc[y1: y1 + h1, x1: x1 + w1])
		if cc_region.size == 0 :
			continue
		#cv2.imshow('cc before', image_resize(cc_region, width = 256))
		img_region = np.ascontiguousarray(img_np[y1: y1 + h1, x1: x1 + w1])
		#cv2.imshow('img', image_resize(img_region, width = 256))
		cc_region = refine_mask(img_region, cc_region)
		#cv2.imshow('cc after', image_resize(cc_region, width = 256))
		#cv2.waitKey(0)
		cc[y1: y1 + h1, x1: x1 + w1] = cc_region
		cc = cv2.dilate(cc, kern)
		final_mask = cv2.bitwise_or(final_mask, cc)
	kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# for (x, y, w, h) in text_lines :
	# 	final_mask = cv2.rectangle(final_mask, (x, y), (x + w, y + h), (255), -1)
	return cv2.dilate(final_mask, kern)

def unsharp(image) :
	gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
	return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
