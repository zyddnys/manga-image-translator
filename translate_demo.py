
from functools import reduce
from typing import List, Set, Tuple
from networkx.algorithms.distance_measures import center
import torch
from DBNet_resnet34 import TextDetection
from model_ocr_48px import OCR
from inpainting_aot import AOTGenerator
import einops
import argparse
import imgproc
import cv2
import numpy as np
import craft_utils
import dbnet_utils
import itertools
import networkx as nx
import math
import requests
import os
from collections import Counter
from oscrypto import util as crypto_utils
from utils import BBox, Quadrilateral, image_resize, quadrilateral_can_merge_region, quadrilateral_can_merge_region_coarse

parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--mode', default='demo', type=str, help='Run demo in either single image demo mode (demo) or web service mode (web)')
parser.add_argument('--image', default='', type=str, help='Image file if using demo mode')
parser.add_argument('--size', default=1536, type=int, help='image square size')
parser.add_argument('--use-inpainting', action='store_true', help='turn on/off inpainting')
parser.add_argument('--use-cuda', action='store_true', help='turn on/off cuda')
parser.add_argument('--inpainting-size', default=2048, type=int, help='size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.0, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='text rendering magnification ratio, larger means higher quality')
args = parser.parse_args()
print(args)

import unicodedata

def _is_whitespace(ch):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically contorl characters but we treat them
	# as whitespace since they are generally considered as such.
	if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
		return True
	cat = unicodedata.category(ch)
	if cat == "Zs":
		return True
	return False


def _is_control(ch):
	"""Checks whether `chars` is a control character."""
	"""Checks whether `chars` is a whitespace character."""
	# These are technically control characters but we count them as whitespace
	# characters.
	if ch == "\t" or ch == "\n" or ch == "\r":
		return False
	cat = unicodedata.category(ch)
	if cat in ("Cc", "Cf"):
		return True
	return False


def _is_punctuation(ch):
	"""Checks whether `chars` is a punctuation character."""
	"""Checks whether `chars` is a whitespace character."""
	cp = ord(ch)
	# We treat all non-letter/number ASCII as punctuation.
	# Characters such as "^", "$", and "`" are not in the Unicode
	# Punctuation class but we treat them as punctuation anyways, for
	# consistency.
	if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
		(cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
		return True
	cat = unicodedata.category(ch)
	if cat.startswith("P"):
		return True
	return False

def count_valuable_text(text) :
	return sum([1 for ch in text if not _is_punctuation(ch) and not _is_control(ch) and not _is_whitespace(ch)])

def bbox_direction(x1, y1, w1, h1, ratio = 2.2) :
	if w1 > h1 * ratio :
		return 'h'
	else :
		return 'v'

def can_merge_textline(x1, y1, w1, h1, x2, y2, w2, h2, ratio = 2.2, char_diff_ratio = 0.5, char_gap_tolerance = 0.5) :
	char_size = min(h1, h2, w1, w2)
	if w1 > h1 * ratio and w2 > h2 * ratio : # both horizontal
		char_size = min(h1, h2)
		if abs(h1 - h2) > char_size * char_diff_ratio :
			return False
		if abs(y1 - y2) > char_size * char_diff_ratio :
			return False
		if x1 < x2 :
			if abs(x1 + w1 - x2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(x2 + w2 - x1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	elif h1 > w1 * ratio and h2 > w2 * ratio : # both vertical
		char_size = min(w1, w2)
		if abs(w1 - w2) > char_size * char_diff_ratio :
			return False
		if abs(x1 - x2) > char_size * char_diff_ratio :
			return False
		if y1 < y2 :
			if abs(y1 + h1 - y2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(y2 + h2 - y1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	elif h1 > w1 * ratio : # box 1 is vertical
		char_size = w1
		if abs(w1 - w2) > char_size * char_diff_ratio :
			return False
		if abs(x1 - x2) > char_size * char_diff_ratio :
			return False
		if y1 < y2 :
			if abs(y1 + h1 - y2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(y2 + h2 - y1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	elif w1 > h1 * ratio : # box 1 is horizontal
		char_size = h1
		if abs(h1 - h2) > char_size * char_diff_ratio :
			return False
		if abs(y1 - y2) > char_size * char_diff_ratio :
			return False
		if x1 < x2 :
			if abs(x1 + w1 - x2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(x2 + w2 - x1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	elif h2 > w2 * ratio : # box 2 is vertical
		char_size = w2
		if abs(w1 - w2) > char_size * char_diff_ratio :
			return False
		if abs(x1 - x2) > char_size * char_diff_ratio :
			return False
		if y1 < y2 :
			if abs(y1 + h1 - y2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(y2 + h2 - y1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	elif w2 > h2 * ratio : # box 2 is horizontal
		char_size = h2
		if abs(h1 - h2) > char_size * char_diff_ratio :
			return False
		if abs(y1 - y2) > char_size * char_diff_ratio :
			return False
		if x1 < x2 :
			if abs(x1 + w1 - x2) > char_size * char_gap_tolerance :
				return False
			else :
				return True
		else :
			if abs(x2 + w2 - x1) > char_size * char_gap_tolerance :
				return False
			else :
				return True
	return False

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

def can_merge_text_region(x1, y1, w1, h1, x2, y2, w2, h2, ratio = 1.9, char_gap_tolerance = 0.9, char_gap_tolerance2 = 1.5) :
	dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
	char_size = min(h1, h2, w1, w2)
	if dist < char_size * char_gap_tolerance :
		if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2 :
			return True
		if w1 > h1 * ratio and h2 > w2 * ratio :
			return False
		if w2 > h2 * ratio and h1 > w1 * ratio :
			return False
		if w1 > h1 * ratio or w2 > h2 * ratio : # h
			return abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
		elif h1 > w1 * ratio or h2 > w2 * ratio : # v
			return abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
		return False
	else :
		return False
	
def get_mini_boxes(contour):
	bounding_box = cv2.minAreaRect(contour)
	points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

	index_1, index_2, index_3, index_4 = 0, 1, 2, 3
	if points[1][1] > points[0][1]:
		index_1 = 0
		index_4 = 1
	else:
		index_1 = 1
		index_4 = 0
	if points[3][1] > points[2][1]:
		index_2 = 2
		index_3 = 3
	else:
		index_2 = 3
		index_3 = 2

	box = [points[index_1], points[index_2], points[index_3], points[index_4]]
	box = np.array(box)
	startidx = box.sum(axis=1).argmin()
	box = np.roll(box, 4-startidx, 0)
	box = np.array(box)
	return box

def split_text_region(bboxes: List[Quadrilateral], region_indices: Set[int], gamma = 0.5, sigma = 2, std_threshold = 6.0) -> List[Set[int]] :
	region_indices = list(region_indices)
	#print('to split', region_indices)
	if len(region_indices) == 1 :
		# case #1
		return [set(region_indices)]
	if len(region_indices) == 2 :
		# case #2
		fs1 = bboxes[region_indices[0]].font_size
		fs2 = bboxes[region_indices[1]].font_size
		fs = max(fs1, fs2)
		if bboxes[region_indices[0]].distance(bboxes[region_indices[1]]) < (1 + gamma) * fs \
			and abs(bboxes[region_indices[0]].angle - bboxes[region_indices[1]].angle) < 4 * np.pi / 180 :
			return [set(region_indices)]
		else :
			return [set([region_indices[0]]), set([region_indices[1]])]
	# case 3
	G = nx.Graph()
	for idx in region_indices :
		G.add_node(idx)
	for (u, v) in itertools.combinations(region_indices, 2) :
		G.add_edge(u, v, weight = bboxes[u].distance(bboxes[v]))
	edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm = "kruskal", data = True)
	edges = sorted(edges, key = lambda a: a[2]['weight'], reverse = True)
	edge_weights = [a[2]['weight'] for a in edges]
	fontsize = np.mean([bboxes[idx].font_size for idx in region_indices])
	std = np.std(edge_weights)
	mean = np.mean(edge_weights)
	#print('edge_weights', edge_weights)
	#print(f'std: {std}, mean: {mean}')
	if (edge_weights[0] <= mean + std * sigma or edge_weights[0] <= fontsize * (1 + gamma)) and std < std_threshold :
		return [set(region_indices)]
	else :
		if edge_weights[0] - edge_weights[1] < std * sigma and std < std_threshold :
			return [set(region_indices)]
		(split_u, split_v, _) = edges[0]
		#print(f'split between "{bboxes[split_u].text}", "{bboxes[split_v].text}"')
		G = nx.Graph()
		for idx in region_indices :
			G.add_node(idx)
		for edge in edges[1:] :
			G.add_edge(edge[0], edge[1])
		ans = []
		for node_set in nx.algorithms.components.connected_components(G) :
			ans.extend(split_text_region(bboxes, node_set))
		return ans
	pass

def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height) :
	G = nx.Graph()
	for i, box in enumerate(bboxes) :
		G.add_node(i, box = box)
		bboxes[i].assigned_index = i
	# step 1: roughly divide into multiple text region candidates
	for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
		if quadrilateral_can_merge_region_coarse(ubox, vbox) :
			G.add_edge(u, v)

	region_indices: List[Set[int]] = []
	for node_set in nx.algorithms.components.connected_components(G) :
		# step 2: split each region
		#print(' -- spliting', node_set)
		region_indices.extend(split_text_region(bboxes, node_set))
	#print('region_indices', region_indices)

	for node_set in region_indices :
		nodes = list(node_set)
		# get overall bbox
		txtlns = np.array(bboxes)[nodes]
		kq = np.concatenate([x.pts for x in txtlns], axis = 0)
		if sum([int(a.is_approximate_axis_aligned) for a in txtlns]) > len(txtlns) // 2 :
			max_coord = np.max(kq, axis = 0)
			min_coord = np.min(kq, axis = 0)
			merged_box = np.maximum(np.array([
				np.array([min_coord[0], min_coord[1]]),
				np.array([max_coord[0], min_coord[1]]),
				np.array([max_coord[0], max_coord[1]]),
				np.array([min_coord[0], max_coord[1]])
				]), 0)
			bbox = np.concatenate([a[None, :] for a in merged_box], axis = 0).astype(int)
		else :
			# TODO: use better method
			bbox = np.concatenate([a[None, :] for a in get_mini_boxes(kq)], axis = 0).astype(int)
		# calculate average fg and bg color
		fg_r = round(np.mean([box.fg_r for box in [bboxes[i] for i in nodes]]))
		fg_g = round(np.mean([box.fg_g for box in [bboxes[i] for i in nodes]]))
		fg_b = round(np.mean([box.fg_b for box in [bboxes[i] for i in nodes]]))
		bg_r = round(np.mean([box.bg_r for box in [bboxes[i] for i in nodes]]))
		bg_g = round(np.mean([box.bg_g for box in [bboxes[i] for i in nodes]]))
		bg_b = round(np.mean([box.bg_b for box in [bboxes[i] for i in nodes]]))
		# majority vote for direction
		dirs = [box.direction for box in [bboxes[i] for i in nodes]]
		majority_dir = Counter(dirs).most_common(1)[0][0]
		# sort
		if majority_dir == 'h' :
			nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
		elif majority_dir == 'v' :
			nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
		# yield overall bbox and sorted indices
		yield bbox, nodes, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b

def generate_text_direction(bboxes: List[Quadrilateral]) :
	G = nx.Graph()
	for i, box in enumerate(bboxes) :
		G.add_node(i, box = box)
	for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
		if quadrilateral_can_merge_region(ubox, vbox) :
			G.add_edge(u, v)
	for node_set in nx.algorithms.components.connected_components(G) :
		nodes = list(node_set)
		# majority vote for direction
		dirs = [box.direction for box in [bboxes[i] for i in nodes]]
		majority_dir = Counter(dirs).most_common(1)[0][0]
		# sort
		if majority_dir == 'h' :
			nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
		elif majority_dir == 'v' :
			nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
		# yield overall bbox and sorted indices
		for node in nodes :
			yield bboxes[node], majority_dir

def run_detect(model, img_np_resized) :
	img_np_resized = img_np_resized.astype(np.float32) / 127.5 - 1.0
	img = torch.from_numpy(img_np_resized)
	if args.use_cuda :
		img = img.cuda()
	img = einops.rearrange(img, 'h w c -> 1 c h w')
	with torch.no_grad() :
		db, mask = model(img)
		db = db.sigmoid().cpu()
		mask = mask[0, 0, :, :].cpu().numpy()
	return db, (mask * 255.0).astype(np.uint8)

def overlay_image(a, b, wa = 0.7) :
	return cv2.addWeighted(a, wa, b, 1 - wa, 0)

def overlay_mask(img, mask) :
	img2 = img.copy().astype(np.float32)
	mask_fp32 = (mask > 10).astype(np.uint8) * 2
	mask_fp32[mask_fp32 == 0] = 1
	mask_fp32 = mask_fp32.astype(np.float32) * 0.5
	img2 = img2 * mask_fp32[:, :, None]
	return img2.astype(np.uint8)

from copy import deepcopy

def filter_bbox(polys) :
	r = []
	for ubox in polys :
		x, y, w, h = ubox[0][0], ubox[0][1], ubox[1][0] - ubox[0][0], ubox[2][1] - ubox[1][1]
		if w / h > 2.5 or h / w > 2.5 :
			r.append(ubox)
	return np.array(r)

def test_inference(img, model) :
	with torch.no_grad() :
		char_probs, prob = model.infer_beam(img, beams_k = 5, max_seq_length = 127)
		_, pred_chars_index = char_probs.max(2)
		pred_chars_index = pred_chars_index.squeeze_(0)
		return pred_chars_index, prob

def ocr_infer_bacth(img, model, widths) :
	if args.use_cuda :
		img = img.cuda()
	with torch.no_grad() :
		return model.infer_beam_batch(img, widths, beams_k = 5, max_seq_length = 255)

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def run_ocr(img, quadrilaterals: List[Tuple[Quadrilateral, str]], dictionary, model, max_chunk_size = 2) :
	text_height = 48
	regions = [q.get_transformed_region(img, d, text_height) for q, d in quadrilaterals]
	out_regions = []
	perm = sorted(range(len(regions)), key = lambda x: regions[x].shape[1])
	ix = 0
	for indices in chunks(perm, max_chunk_size) :
		N = len(indices)
		widths = [regions[i].shape[1] for i in indices]
		max_width = 4 * (max(widths) + 7) // 4
		region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
		for i, idx in enumerate(indices) :
			W = regions[idx].shape[1]
			region[i, :, : W, :] = regions[idx]
			cv2.imwrite(f'ocrs/{ix}.png', region[i, :, :, :])
			ix += 1
		images = (torch.from_numpy(region).float() - 127.5) / 127.5
		images = einops.rearrange(images, 'N H W C -> N C H W')
		ret = ocr_infer_bacth(images, model, widths)
		for i, (pred_chars_index, prob, fr, fg, fb, br, bg, bb) in enumerate(ret) :
			if prob < 0.4 :
				continue
			fr = (torch.clip(fr.view(-1), 0, 1).mean() * 255).long().item()
			fg = (torch.clip(fg.view(-1), 0, 1).mean() * 255).long().item()
			fb = (torch.clip(fb.view(-1), 0, 1).mean() * 255).long().item()
			br = (torch.clip(br.view(-1), 0, 1).mean() * 255).long().item()
			bg = (torch.clip(bg.view(-1), 0, 1).mean() * 255).long().item()
			bb = (torch.clip(bb.view(-1), 0, 1).mean() * 255).long().item()
			seq = []
			for chid in pred_chars_index :
				ch = dictionary[chid]
				if ch == '<S>' :
					continue
				if ch == '</S>' :
					break
				if ch == '<SP>' :
					ch = ' '
				seq.append(ch)
			txt = ''.join(seq)
			print(prob, txt, f'fg: ({fr}, {fg}, {fb})', f'bg: ({br}, {bg}, {bb})')
			cur_region = quadrilaterals[indices[i]][0]
			cur_region.text = txt
			cur_region.prob = prob
			cur_region.fg_r = fr
			cur_region.fg_g = fg
			cur_region.fg_b = fb
			cur_region.bg_r = br
			cur_region.bg_g = bg
			cur_region.bg_b = bb
			out_regions.append(cur_region)
	return out_regions

def resize_keep_aspect(img, size) :
	ratio = (float(size)/max(img.shape[0], img.shape[1]))
	new_width = round(img.shape[1] * ratio)
	new_height = round(img.shape[0] * ratio)
	return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR_EXACT)

def run_inpainting(model_inpainting, img, mask, max_image_size = 1024, pad_size = 4) :
	img_original = np.copy(img)
	mask_original = np.copy(mask)
	mask_original[mask_original < 127] = 0
	mask_original[mask_original >= 127] = 1
	mask_original = mask_original[:, :, None]
	if not args.use_inpainting :
		img = np.copy(img)
		img[mask > 0] = np.array([255, 255, 255], np.uint8)
		return img, None
	height, width, c = img.shape
	if max(img.shape[0: 2]) > max_image_size :
		img = resize_keep_aspect(img, max_image_size)
		mask = resize_keep_aspect(mask, max_image_size)
	h, w, c = img.shape
	if h % pad_size != 0 :
		new_h = (pad_size - (h % pad_size)) + h
	else :
		new_h = h
	if w % pad_size != 0 :
		new_w = (pad_size - (w % pad_size)) + w
	else :
		new_w = w
	if new_h != h or new_w != w :
		img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR_EXACT)
		mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_LINEAR_EXACT)
	print(f'Inpainting resolution: {new_w}x{new_h}')
	img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
	mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
	mask_torch[mask_torch < 0.5] = 0
	mask_torch[mask_torch >= 0.5] = 1
	if args.use_cuda :
		img_torch = img_torch.cuda()
		mask_torch = mask_torch.cuda()
	with torch.no_grad() :
		img_torch *= (1 - mask_torch)
		img_inpainted_torch = model_inpainting(img_torch, mask_torch)
	img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
	if new_h != height or new_w != width :
		img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation = cv2.INTER_LINEAR_EXACT)
	return img_inpainted * mask_original + img_original * (1 - mask_original), (img_torch.cpu() * 127.5 + 127.5).squeeze_(0).permute(1, 2, 0).numpy()

import text_render

def load_ocr_model() :
	with open('alphabet-all-v5.txt', 'r', encoding='utf-8') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	model = OCR(dictionary, 768)
	model.load_state_dict(torch.load('ocr_48px.ckpt', map_location='cpu'))
	model.eval()
	if args.use_cuda :
		model = model.cuda()
	return dictionary, model

def load_detect_model() :
	model = TextDetection()
	sd = torch.load('detect.ckpt', map_location='cpu')
	model.load_state_dict(sd['model'] if 'model' in sd else sd)
	model.eval()
	if args.use_cuda :
		model = model.cuda()
	return model

def load_inpainting_model() :
	if not args.use_inpainting :
		return 'not available'
	model = AOTGenerator()
	sd = torch.load('inpainting.ckpt', map_location='cpu')
	model.load_state_dict(sd['gen'] if 'gen' in sd else sd)
	model.eval()
	if args.use_cuda :
		model = model.cuda()
	return model

def update_state(task_id, nonce, state) :
	requests.post('http://127.0.0.1:5003/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state})

def get_task(nonce) :
	try :
		rjson = requests.get(f'http://127.0.0.1:5003/task-internal?nonce={nonce}').json()
		if 'task_id' in rjson :
			return rjson['task_id']
		else :
			return None
	except :
		return None

async def infer(
	img,
	mode,
	nonce,
	dictionary,
	model_detect,
	model_ocr,
	model_inpainting,
	task_id = ''
	) :
	img_detect_size = args.size
	if task_id and len(task_id) != 32 :
		size_ind = task_id[-1]
		if size_ind == 'S' :
			img_detect_size = 1024
		elif size_ind == 'M' :
			img_detect_size = 1536
		elif size_ind == 'L' :
			img_detect_size = 2048
		elif size_ind == 'X' :
			img_detect_size = 2560
		print(f' -- Detection size {size_ind}, resolution {img_detect_size}')
	print(' -- Read image')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_bbox = np.copy(img)
	img_bbox_raw = np.copy(img_bbox)
	img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(cv2.bilateralFilter(img, 17, 80, 80), img_detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
	img_to_overlay = np.copy(img_resized)
	ratio_h = ratio_w = 1 / target_ratio

	print(f'Detection resolution: {img_resized.shape[1]}x{img_resized.shape[0]}')
	print(' -- Running text detection')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'detection')
	db, mask = run_detect(model_detect, img_resized)
	overlay = imgproc.cvt2HeatmapImg(db[0, 0, :, :].numpy())
	det = dbnet_utils.SegDetectorRepresenter(args.text_threshold, args.box_threshold, unclip_ratio = args.unclip_ratio)
	boxes, scores = det({'shape':[(img_resized.shape[0], img_resized.shape[1])]}, db)
	boxes, scores = boxes[0], scores[0]
	if boxes.size == 0 :
		polys = []
	else :
		idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
		polys, _ = boxes[idx], scores[idx]
		polys = polys.astype(np.float64)
		polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 1)
		polys = polys.astype(np.int16)
	textlines = [Quadrilateral(pts.astype(int), '', 0) for pts in polys]
	for txtln in textlines :
		cv2.polylines(img_bbox_raw, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
		[l1a, l1b, l2a, l2b] = txtln.structure
		cv2.line(img_bbox_raw, l1a, l1b, color = (0, 255, 0), thickness = 1)
		cv2.line(img_bbox_raw, l2a, l2b, color = (0, 0, 255), thickness = 1)
		dbox = txtln.aabb
		cv2.rectangle(img_bbox_raw, (dbox.x, dbox.y), (dbox.x + dbox.w, dbox.y + dbox.h), color = (255, 0, 255), thickness = 2)

	print(' -- Running OCR')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'ocr')
	textlines = run_ocr(img_bbox, list(generate_text_direction(textlines)), dictionary, model_ocr, 16)

	text_regions: List[Quadrilateral] = []
	new_textlines = []
	for (poly_regions, textline_indices, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b) in merge_bboxes_text_region(textlines, img_bbox_raw.shape[1], img_bbox_raw.shape[0]) :
		text = ''
		logprob_lengths = []
		for textline_idx in textline_indices :
			if not text :
				text = textlines[textline_idx].text
			else :
				last_ch = text[-1]
				cur_ch = textlines[textline_idx].text[0]
				if ord(last_ch) > 255 and ord(cur_ch) > 255 :
					text += textlines[textline_idx].text
				else :
					if last_ch == '-' and ord(cur_ch) < 255 :
						text = text[:-1] + textlines[textline_idx].text
					else :
						text += ' ' + textlines[textline_idx].text
			logprob_lengths.append((np.log(textlines[textline_idx].prob), len(textlines[textline_idx].text)))
		vc = count_valuable_text(text)
		total_logprobs = 0.0
		for (logprob, length) in logprob_lengths :
			total_logprobs += logprob * length
		total_logprobs /= sum([x[1] for x in logprob_lengths])
		# filter text region without characters
		if vc > 1 :
			region = Quadrilateral(poly_regions, text, np.exp(total_logprobs), fg_r, fg_g, fg_b, bg_r, bg_g, bg_b)
			region.clip(img.shape[1], img.shape[0])
			region.textline_indices = []
			region.majority_dir = majority_dir
			text_regions.append(region)
			for textline_idx in textline_indices :
				region.textline_indices.append(len(new_textlines))
				new_textlines.append(textlines[textline_idx])
	textlines: List[Quadrilateral] = new_textlines

	if mode == 'web' and task_id :
		print(' -- Translating')
		update_state(task_id, nonce, 'translating')
		# in web mode, we can start translation task async
		requests.post('http://127.0.0.1:5003/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.text for r in text_regions]})

	print(' -- Generating text mask')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'mask_generation')
	# create mask
	from text_mask_utils import filter_masks, complete_mask
	mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation = cv2.INTER_LINEAR)
	if pad_h > 0 :
		mask_resized = mask_resized[:-pad_h, :]
	elif pad_w > 0 :
		mask_resized = mask_resized[:, : -pad_w]
	mask_resized = cv2.resize(mask_resized, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
	img_resized_2 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
	mask_resized[mask_resized > 150] = 255
	text_lines = [(a.aabb.x // 2, a.aabb.y // 2, a.aabb.w // 2, a.aabb.h // 2) for a in textlines]
	mask_ccs, cc2textline_assignment = filter_masks(mask_resized, text_lines)
	if mask_ccs :
		mask_filtered = reduce(cv2.bitwise_or, mask_ccs)
		cv2.imwrite(f'result/{task_id}/mask_filtered.png', mask_filtered)
		cv2.imwrite(f'result/{task_id}/mask_filtered_img.png', overlay_mask(img_resized_2, mask_filtered))
		final_mask = complete_mask(img_resized_2, mask_ccs, text_lines, cc2textline_assignment)
		cv2.imwrite(f'result/{task_id}/mask.png', final_mask)
		cv2.imwrite(f'result/{task_id}/mask_img.png', overlay_mask(img_resized_2, final_mask))
		final_mask = cv2.resize(final_mask, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR)
		final_mask[final_mask > 0] = 255
	else :
		final_mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)


	print(' -- Running inpainting')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'inpainting')
	# run inpainting
	img_inpainted, inpaint_input = run_inpainting(model_inpainting, img, final_mask, args.inpainting_size)

	# translate text region texts
	if mode != 'web' :
		print(' -- Translating')
		texts = '\n'.join([r.text for r in text_regions])
		if texts :
			from youdao import Translator
			translator = Translator()
			trans_ret = await translator.translate('auto', 'zh-CHS', texts)
			#trans_ret = [r.text for r in text_regions]
		else :
			trans_ret = []
		if trans_ret :
			translated_sentences = []
			batch = len(text_regions)
			if len(trans_ret) < batch :
				translated_sentences.extend(trans_ret)
				translated_sentences.extend([''] * (batch - len(trans_ret)))
			elif len(trans_ret) > batch :
				translated_sentences.extend(trans_ret[:batch])
			else :
				translated_sentences.extend(trans_ret)
		else :
			translated_sentences = texts
	else :
		# wait for at most 1 hour
		translated_sentences = None
		for _ in range(36000) :
			ret = requests.post('http://127.0.0.1:5003/get-translation-result-internal', json = {'task_id': task_id, 'nonce': nonce}).json()
			if 'result' in ret :
				translated_sentences = ret['result']
				break
			await asyncio.sleep(0.1)
	if not translated_sentences and text_regions :
		update_state(task_id, nonce, 'error')
		return

	print(' -- Rendering translated text')
	# render translated texts
	img_canvas = np.copy(img_inpainted)
	from utils import findNextPowerOf2
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		if not trans_text :
			continue
		print(region.text)
		print(trans_text)
		#print(region.majority_dir, region.pts)
		fg = (region.fg_r, region.fg_g, region.fg_b)
		bg = (region.bg_r, region.bg_g, region.bg_b)
		font_size = 0
		n_lines = len(region.textline_indices)
		for idx in region.textline_indices :
			txtln = textlines[idx]
			img_bbox = cv2.polylines(img_bbox, [txtln.pts], True, color = fg, thickness=2)
			# [l1a, l1b, l2a, l2b] = txtln.structure
			# cv2.line(img_bbox, l1a, l1b, color = (0, 255, 0), thickness = 2)
			# cv2.line(img_bbox, l2a, l2b, color = (0, 0, 255), thickness = 2)
			dbox = txtln.aabb
			font_size = max(font_size, txtln.font_size)
			#cv2.rectangle(img_bbox, (dbox.x, dbox.y), (dbox.x + dbox.w, dbox.y + dbox.h), color = (255, 0, 255), thickness = 2)
		font_size = round(font_size)
		img_bbox = cv2.polylines(img_bbox, [region.pts], True, color=(0, 0, 255), thickness = 2)

		region_aabb = region.aabb
		print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)

		# round font_size to fixed powers of 2, so later LRU cache can work
		font_size_enlarged = findNextPowerOf2(font_size) * args.text_mag_ratio
		enlarge_ratio = font_size_enlarged / font_size
		font_size = font_size_enlarged
		while True :
			enlarged_w = round(enlarge_ratio * region_aabb.w)
			enlarged_h = round(enlarge_ratio * region_aabb.h)
			rows = enlarged_h // (font_size * 1.3)
			cols = enlarged_w // (font_size * 1.3)
			if rows * cols < len(trans_text) :
				enlarge_ratio *= 1.1
				continue
			break
		print('font_size:', font_size)

		tmp_canvas = np.ones((enlarged_h * 2, enlarged_w * 2, 3), dtype = np.uint8) * 127
		tmp_mask = np.zeros((enlarged_h * 2, enlarged_w * 2), dtype = np.uint16)

		if region.majority_dir == 'h' :
			text_render.put_text_horizontal(
				font_size,
				enlarge_ratio * 1.0,
				tmp_canvas,
				tmp_mask,
				trans_text,
				len(region.textline_indices),
				[textlines[idx] for idx in region.textline_indices],
				enlarged_w // 2,
				enlarged_h // 2,
				enlarged_w,
				enlarged_h,
				fg,
				bg
			)
		else :
			text_render.put_text_vertical(
				font_size,
				enlarge_ratio * 1.0,
				tmp_canvas,
				tmp_mask,
				trans_text,
				len(region.textline_indices),
				[textlines[idx] for idx in region.textline_indices],
				enlarged_w // 2,
				enlarged_h // 2,
				enlarged_w,
				enlarged_h,
				fg,
				bg
			)

		tmp_mask = np.clip(tmp_mask, 0, 255).astype(np.uint8)
		x, y, w, h = cv2.boundingRect(tmp_mask)
		r_prime = w / h
		r = region.aspect_ratio
		w_ext = 0
		h_ext = 0
		if r_prime > r :
			h_ext = w / (2 * r) - h / 2
		else :
			w_ext = (h * r - w) / 2
		region_ext = round(min(w, h) * 0.05)
		h_ext += region_ext
		w_ext += region_ext
		src_pts = np.array([[x - w_ext, y - h_ext], [x + w + w_ext, y - h_ext], [x + w + w_ext, y + h + h_ext], [x - w_ext, y + h + h_ext]]).astype(np.float32)
		src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
		src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)
		dst_pts = region.pts
		M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		tmp_rgba = np.concatenate([tmp_canvas, tmp_mask[:, :, None]], axis = -1).astype(np.float32)
		rgba_region = np.clip(cv2.warpPerspective(tmp_rgba, M, (img_canvas.shape[1], img_canvas.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
		canvas_region = rgba_region[:, :, 0: 3]
		mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
		img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)

	print(' -- Saving results')
	result_db = db[0, 0, :, :].numpy()
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	cv2.imwrite(f'result/{task_id}/db.png', imgproc.cvt2HeatmapImg(result_db))
	cv2.imwrite(f'result/{task_id}/textline.png', overlay)
	cv2.imwrite(f'result/{task_id}/bbox.png', cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))
	cv2.imwrite(f'result/{task_id}/bbox_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))
	cv2.imwrite(f'result/{task_id}/overlay.png', cv2.cvtColor(overlay_image(img_to_overlay, cv2.resize(overlay, (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_LINEAR)), cv2.COLOR_RGB2BGR))
	cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))
	if inpaint_input is not None :
		cv2.imwrite(f'result/{task_id}/inpaint_input.png', cv2.cvtColor(inpaint_input, cv2.COLOR_RGB2BGR))
	cv2.imwrite(f'result/{task_id}/final.png', cv2.cvtColor(img_canvas, cv2.COLOR_RGB2BGR))

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'finished')

from PIL import Image
import time
import asyncio

async def main(mode = 'demo') :
	print(' -- Loading models')
	import os
	os.makedirs('result', exist_ok = True)
	text_render.prepare_renderer()
	dictionary, model_ocr = load_ocr_model()
	model_detect = load_detect_model()
	model_inpainting = load_inpainting_model()

	if mode == 'demo' :
		print(' -- Running in single image demo mode')
		if not args.image :
			print('please provide an image')
			parser.print_usage()
			return
		img = cv2.imread(args.image)
		await infer(img, mode, '', dictionary, model_detect, model_ocr, model_inpainting)
	elif mode == 'web' :
		print(' -- Running in web service mode')
		print(' -- Waiting for translation tasks')
		nonce = crypto_utils.rand_bytes(16).hex()
		import subprocess
		import sys
		subprocess.Popen([sys.executable, 'web_main.py', nonce, '5003'])
		while True :
			task_id = get_task(nonce)
			if task_id :
				print(f' -- Processing task {task_id}')
				img = cv2.imread(f'result/{task_id}/input.png')
				try :
					infer_task = asyncio.create_task(infer(img, mode, nonce, dictionary, model_detect, model_ocr, model_inpainting, task_id))
					asyncio.gather(infer_task)
				except :
					import traceback
					traceback.print_exc()
					update_state(task_id, nonce, 'error')
			else :
				await asyncio.sleep(0.1)
	

if __name__ == '__main__':
	asyncio.run(main(args.mode))
