
from typing import Tuple, List
import numpy as np
import cv2
import math

from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from functools import reduce
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import argparse
parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--image', default='', type=str, help='Image file')
args = parser.parse_args()

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
	# kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# mask_erode = ((cv2.erode(mask_img, kern2) > 0) * 255).astype(np.uint8)
	# save_rgb('text_mask_utils_raw_mask_erode.png', mask_erode)
	for (x, y, w, h) in text_lines :
		cv2.rectangle(mask_img, (x, y), (x + w, y + h), (0), 1)
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img)
	
	# Map component labels to hue val, 0-179 is the hue range in OpenCV
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# Converting cvt to BGR
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue==0] = 0


	cc2textline_assignment = []

	#cv2.imwrite('text_mask_utils_cc.png', labeled_img)
	result = []
	kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# for i in range(1, num_labels) :
	# 	if stats[i, cv2.CC_STAT_AREA] <= 9 :
	# 		continue # skip area too small
	# 	cc = np.zeros_like(mask_img)
	# 	cc[labels == i] = 255
	# 	x1, y1, w1, h1 = cv2.boundingRect(cc)
	# 	area1 = w1 * h1
	# 	for j, (x2, y2, w2, h2) in enumerate(text_lines) :
	# 		area2 = w2 * h2
	# 		overlapping_area = area(x1, y1, w1, h1, x2, y2, w2, h2)
	# 		ratio = overlapping_area / min(area1, area2)
	# 		if ratio > keep_threshold :
	# 			#result.append(cv2.dilate(cc, kern))
	# 			result.append(cv2.erode(cc, kern))
	# 			cc2textline_assignment.append(j)
	# 			#result.append(np.copy(cc))
	# 			break
	M = len(text_lines)
	ratio_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
	dist_mat = np.zeros(shape = (num_labels, M), dtype = np.float32)
	for i in range(1, num_labels) :
		if stats[i, cv2.CC_STAT_AREA] <= 9 :
			continue # skip area too small
		cc = np.zeros_like(mask_img)
		cc[labels == i] = 255
		#cc = cv2.dilate(cc, kern2)
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
			#result.append(cv2.dilate(cc, kern))
			#result.append(cv2.erode(cc, kern))
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
	result_img = reduce(cv2.bitwise_or, result)
		
	#cv2.imwrite('text_mask_utils_filtered_mask.png', result_img)
	return result, cc2textline_assignment

def find_top_k_dpgmm(result: BayesianGaussianMixture, k: int = 2, cluster_threshold = 0.5) :
	total_energy = np.sum(result.weights_)
	num_clusters = result.n_components
	perm = sorted(range(result.n_components), key=lambda k: -result.weights_[k])
	if (result.covariances_[perm[:k]] > 100).any() :
		perm = sorted(range(result.n_components), key=lambda k: result.covariances_[k].mean())
	prefix_sum = 0
	for i in range(result.n_components) :
		prefix_sum += result.weights_[perm[i]]
		if prefix_sum > total_energy * cluster_threshold :
			num_clusters = i + 1
			break
	return np.round(result.means_[perm[:k]]).astype(np.int16), np.round(np.sqrt(result.covariances_[perm[:k]])).astype(np.int16), num_clusters

def inRangeMasked(img, mask, lowb, upb) :
	if lowb[0] > 0 :
		masked_r = 0
	elif upb[0] < 255 :
		masked_r = 255
	else :
		raise Exception('R ranges entire uint8 space')
	if lowb[1] > 0 :
		masked_g = 0
	elif upb[1] < 255 :
		masked_g = 255
	else :
		raise Exception('G ranges entire uint8 space')
	if lowb[2] > 0 :
		masked_b = 0
	elif upb[2] < 255 :
		masked_b = 255
	else :
		raise Exception('B ranges entire uint8 space')
	img2 = np.copy(img)
	img2[mask < 127] = np.array([masked_r, masked_g, masked_b], dtype = np.uint8)
	return cv2.inRange(img2, lowb, upb)

def extend_cc_region(color_image: np.ndarray, cc: np.ndarray, color, std_ext) :
	x, y, w, h = cv2.boundingRect(cc)
	cc_region = color_image[y: y + h, x: x + w]
	cc_region_mask = cc[y: y + h, x: x + w]
	#seed_point_candidates_mask = cv2.inRange(cc_region, np.clip(color - std_ext, 0, 255).astype(np.uint8), np.clip(color + std_ext, 0, 255).astype(np.uint8))
	lowb = np.clip(color - std_ext, 0, 255).astype(np.uint8)
	upb = np.clip(color + std_ext, 0, 255).astype(np.uint8)
	seed_point_candidates_mask = inRangeMasked(cc_region, cc_region_mask, lowb, upb)
	num_c, labels = cv2.connectedComponents(seed_point_candidates_mask)
	final_mask = np.zeros((cc.shape[0] + 2, cc.shape[1] + 2), dtype = np.uint8)
	for i in range(1, num_c) :
		seed_point_candidates_y, seed_point_candidates_x = np.where(labels == i)
		seed_point = (seed_point_candidates_x[0] + x, seed_point_candidates_y[0] + y)
		seed_color = color_image[seed_point[::-1]]
		# lowb_diff = np.maximum(seed_color.astype(np.int16) - lowb.astype(np.int16), std_ext * 2)
		# upb_diff = np.maximum(upb.astype(np.int16) - seed_color.astype(np.int16), std_ext * 2)
		diff = np.maximum(seed_color.astype(np.int16) - lowb.astype(np.int16), upb.astype(np.int16) - seed_color.astype(np.int16))
		#print(color_image[seed_point[1] - 4: seed_point[1] + 5, seed_point[0] - 4: seed_point[0] + 5][:,:,1])
		#print(seed_color, diff, std_ext)
		cv2.floodFill(color_image, final_mask, seed_point, (255), diff.tolist(), diff.tolist(), cv2.FLOODFILL_MASK_ONLY | 8)
	# if final_mask[1: -1, 1: -1].sum() == 0 :
	# 	breakpoint()
	return final_mask[1: -1, 1: -1] * 255

def main_process(img_np: np.ndarray, ccs: List[np.ndarray], text_lines: List[Tuple[int, int, int, int]], cc2textline_assignment) :
	if len(ccs) == 0 :
		return
	final_mask = np.zeros_like(ccs[0])
	dpgmm = BayesianGaussianMixture(n_components = 5, covariance_type = 'diag')
	kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	text_line_colors = defaultdict(list)
	#print(cc2textline_assignment)
	for i, cc in enumerate(tqdm(ccs)) :
		if np.sum(cv2.bitwise_and(cc, final_mask)) > 0 :
			final_mask = cv2.bitwise_or(final_mask, cc)
			continue
		pixels = img_np[cv2.erode(cc, kern) > 127]#img_np[cc > 127]#
		if len(pixels) < 5 :
			final_mask = cv2.bitwise_or(final_mask, cc)
			continue
		cls1 = dpgmm.fit(pixels)
		# print(cls1.means_)
		# print(cls1.weights_)
		# print(np.sqrt(cls1.covariances_))
		cls1_top2_mean, cls1_top2_stddev, cls1_k = find_top_k_dpgmm(cls1, 2)
		# cls1_top2_mean[0][0] = 4
		# cls1_top2_mean[0][1] = 4
		# cls1_top2_mean[0][2] = 4
		# cls1_top2_mean[1][0] = 253
		# cls1_top2_mean[1][1] = 253
		# cls1_top2_mean[1][2] = 253
		# cls1_top2_stddev[0][0] = 5
		# cls1_top2_stddev[0][1] = 5
		# cls1_top2_stddev[0][2] = 5
		# cls1_top2_stddev[1][0] = 5
		# cls1_top2_stddev[1][1] = 5
		# cls1_top2_stddev[1][2] = 5
		# if cls1_top2_mean[0][0] > 100 :
		# 	breakpoint()
		cls1_top2_stddev_ext = np.round(cls1_top2_stddev * COLOR_RANGE_SIGMA)
		# if i == 2 :
		# 	breakpoint()
		top1_mask = extend_cc_region(img_np, cc, cls1_top2_mean[0], cls1_top2_stddev_ext[0])
		top2_mask = extend_cc_region(img_np, cc, cls1_top2_mean[1], cls1_top2_stddev_ext[1])
		# area1 = int(top1_mask.sum())
		# area2 = int(top2_mask.sum())
		# if area1 == 0 or area2 == 0 :
		# 	breakpoint()
		# 	continue
		# area_cc = int(cc.sum())
		# if abs(area1 - area_cc) < abs(area2 - area_cc) :
		# 	D = top1_mask
		# 	selected_idx = 0
		# else :
		# 	D = top2_mask
		# 	selected_idx = 1
		# intersect_area1 = int(cv2.bitwise_and(cc, top1_mask).sum())
		# intersect_area2 = int(cv2.bitwise_and(cc, top2_mask).sum())
		iou1 = cv2.bitwise_and(cc, top1_mask).sum() / cv2.bitwise_or(cc, top1_mask).sum()
		iou2 = cv2.bitwise_and(cc, top2_mask).sum() / cv2.bitwise_or(cc, top2_mask).sum()
		if iou1 > iou2 :
			D = top1_mask
			selected_idx = 0
			if iou1 < 1e-1 :
				#print(iou1)
				D = cc
				selected_idx = -1
		else :
			D = top2_mask
			selected_idx = 1
			if iou2 < 1e-1 :
				#print(iou2)
				D = cc
				selected_idx = -1
		# print(selected_idx, iou1, iou2)
		# save_rgb('text_mask_utils_tmp.png', cc)
		# save_rgb('text_mask_utils_tmp_color.png', img_np*(cc>0)[:,:,None])
		# save_rgb('text_mask_utils_tmp_top2_mask.png', top2_mask)
		# save_rgb('text_mask_utils_tmp_top1_mask.png', top1_mask)
		# input('x')
		# if cc2textline_assignment[i] == 12 :
		# 	breakpoint()

		D = cv2.bitwise_or(cc, D)
		D = cv2.dilate(D, kern)
		# if cls1_top2_mean[selected_idx][0] < 100 :
		# 	breakpoint()
		final_mask = cv2.bitwise_or(final_mask, D)
		# seed_point_candidates_mask = cv2.inRange(cc_region, cls1_top2_mean[0] - cls1_top2_stddev_ext[0], cls1_top2_mean[0] + cls1_top2_stddev_ext[0])
		# seed_point_candidates_y, seed_point_candidates_x = np.where(seed_point_candidates_mask > 127)
		# seed_point = (seed_point_candidates_x[0] + x, seed_point_candidates_y[0] + y)
		# D = np.zeros((cc.shape[0] + 2, cc.shape[1] + 2), dtype = np.uint8)
		# cv2.floodFill(img_np, D, seed_point, (255), cls1_top2_stddev_ext[0].tolist(), cls1_top2_stddev_ext[0].tolist(), cv2.FLOODFILL_MASK_ONLY)
		# D = D[1: -1, 1: -1] * 255
		# final_mask = cv2.bitwise_or(final_mask, D)

		# now we find text color
		if selected_idx == -1 :
			continue # skip
		text_color_value = cls1_top2_mean[selected_idx]
		text_color_stddev = cls1_top2_stddev[selected_idx]
		#print('color=', text_color_value, text_color_stddev)
		text_line_colors[cc2textline_assignment[i]].append(text_color_value)
	
	textline_img = np.copy(img_np)

	def get_textline_color(j, visited_text_lines: List[int]) :
		visited_text_lines.append(j)
		(x, y, w, h) = text_lines[j]
		colors = text_line_colors[j]
		if not colors :
			#print(f'[!!!!!!!!!!!!] Textline {j} has no color assigned, seraching for closest')
			min_dist = 10000000000000000000
			min_dist_k = -1
			for k, (x2, y2, w2, h2) in enumerate(text_lines) :
				d = rect_distance(x, y, x + w, y + h, x2, y2, x2 + w2, y2 + h2)
				if d < min_dist and k not in visited_text_lines :
					min_dist = d
					min_dist_k = k
			if min_dist_k == -1 :
				print(f'[!!!!!!!!!!!!] Textline {j} has no color assigned and unable to find closest rectangle, defaulting to black')
				return np.zeros((3,), dtype = np.uint8)
			return get_textline_color(min_dist_k, visited_text_lines)
		else :
			clr = np.round(np.mean(np.array(colors), axis = 0))
			return clr

	textline_colors = []
	for j, (x, y, w, h) in enumerate(text_lines) :
		# colors = text_line_colors[j]
		# if not colors :
		# 	print(f'[!!!!!!!!!!!!] Textline {j} has no color assigned')
		# 	cv2.rectangle(textline_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# 	continue
		clr = get_textline_color(j, []).tolist()
		textline_colors.append(clr)
		cv2.rectangle(textline_img, (x, y), (x + w, y + h), clr, 2)
		
	#save_rgb('text_mask_utils_final_textlines.png', textline_img)

	#save_rgb('text_mask_utils_final_mask_masked.png', (final_mask > 127)[:,:,None] * img_np)
	#cv2.imwrite('text_mask_utils_final_mask.png', final_mask)
	return final_mask, textline_colors

def unsharp(image) :
	gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
	return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

def test() :
	img = args.image
	text_lines, mask, img_np = extract_textlines_and_mask(img)
	# for _ in range(1) :
	# 	img_blur = cv2.bilateralFilter(img_np, 7, 20, 20)
	# 	img_blur2 = cv2.GaussianBlur(img_np, (0, 0), 2.0)
	# 	img_np = cv2.addWeighted(img_blur, 1.6, img_blur2, -0.6, 0, img_np)
	save_rgb('text_mask_utils_img_np.png', img_np)
	save_rgb('text_mask_utils_raw_mask.png', mask)
	filtered_dilated_ccs, cc2textline_assignment = filter_masks(mask, text_lines)
	main_process(img_np, filtered_dilated_ccs, text_lines, cc2textline_assignment)


def test_crf() :
	from test_craft import extract_textlines_and_mask
	img = 'test_images/irl19.jpg'
	text_lines, mask, img_np = extract_textlines_and_mask(img)
	save_rgb('text_mask_utils_img_np.png', img_np)
	filtered_dilated_ccs = filter_masks(mask, text_lines)
	from functools import reduce
	new_mask = reduce(lambda x, y: cv2.bitwise_or(x, y), filtered_dilated_ccs)
	#save_rgb('text_mask_utils_img_np.png', img_np)
	import pydensecrf.densecrf as dcrf
	from pydensecrf.utils import compute_unary, unary_from_softmax
	mask_softmax = np.concatenate([cv2.bitwise_not(new_mask)[:,:,None], new_mask[:,:,None]], axis=2)
	mask_softmax = mask_softmax.astype(np.float32) / 255.0

	n_classes = 2
	feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
	print(feat_first.shape)
	unary = unary_from_softmax(feat_first)
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], n_classes)

	d.setUnaryEnergy(unary)
	d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)

	d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=img_np,
						compat=10,
						kernel=dcrf.DIAG_KERNEL,
						normalization=dcrf.NORMALIZE_SYMMETRIC)
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((img_np.shape[0], img_np.shape[1]))
	print(res.shape)
	print(np.unique(res))
	crf_mask = np.array(res*255, dtype=np.uint8)
	save_rgb('text_mask_utils_mask_crf.png', crf_mask)
	save_rgb('text_mask_utils_mask_crf_masked.png', (crf_mask > 127)[:,:,None] * img_np)

if __name__ == '__main__' :
	test()
