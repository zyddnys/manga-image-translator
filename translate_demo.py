
from functools import reduce
from typing import List
from networkx.algorithms.distance_measures import center
import torch
from CRAFT_resnet34 import CRAFT_net
from model_ocr import OCR
from inpainting_model import InpaintingVanilla
import einops
import argparse
import imgproc
import cv2
import numpy as np
import craft_utils
import itertools
import networkx as nx
import math
from collections import Counter

parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--image', default='', type=str, help='Image file')
parser.add_argument('--size', default=1536, type=int, help='image square size')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text_threshold')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link_threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='low_text')
args = parser.parse_args()

import unicodedata

TEXT_EXT_RATIO = 0.1

class BBox(object) :
	def __init__(self, x: int, y: int, w: int, h: int, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0) :
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.text = text
		self.prob = prob
		self.fg_r = fg_r
		self.fg_g = fg_g
		self.fg_b = fg_b
		self.bg_r = bg_r
		self.bg_g = bg_g
		self.bg_b = bg_b

	def to_points(self) :
		tl, tr, br, bl = np.array([self.x, self.y]), np.array([self.x + self.w, self.y]), np.array([self.x + self.w, self.y+ self.h]), np.array([self.x, self.y + self.h])
		return tl, tr, br, bl

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

def can_merge_textline(x1, y1, w1, h1, x2, y2, w2, h2, ratio = 2.2, char_diff_ratio = 0.7, char_gap_tolerance = 1.3) :
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

def can_merge_text_region(x1, y1, w1, h1, x2, y2, w2, h2, ratio = 1.9, char_gap_tolerance = 0.9, char_gap_tolerance2 = 2.5) :
	dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
	char_size = min(h1, h2, w1, w2)
	if dist < char_size * char_gap_tolerance :
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

def merge_bboxes(bboxes, merge_func) :
	G = nx.Graph()
	for i, box in enumerate(bboxes) :
		G.add_node(i, box = box)
	for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
		if merge_func(ubox[0][0], ubox[0][1], ubox[1][0] - ubox[0][0], ubox[2][1] - ubox[1][1], vbox[0][0], vbox[0][1], vbox[1][0] - vbox[0][0], vbox[2][1] - vbox[1][1]) :
			G.add_edge(u, v)
	merged_boxes = []
	for node_set in nx.algorithms.components.connected_components(G) :
		kq = np.concatenate(bboxes[list(node_set)], axis = 0)
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		merged_boxes.append(np.maximum(np.array([
			np.array([min_coord[0], min_coord[1]]),
			np.array([max_coord[0], min_coord[1]]),
			np.array([max_coord[0], max_coord[1]]),
			np.array([min_coord[0], max_coord[1]])
			]), 0))
	return np.array(merged_boxes)
	
def merge_bboxes_text_region(bboxes: List[BBox]) :
	G = nx.Graph()
	for i, box in enumerate(bboxes) :
		G.add_node(i, box = box)
	for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
		if can_merge_text_region(ubox.x, ubox.y, ubox.w, ubox.h, vbox.x, vbox.y, vbox.w, vbox.h) :
			G.add_edge(u, v)
	for node_set in nx.algorithms.components.connected_components(G) :
		nodes = list(node_set)
		# get overall bbox
		kq = np.concatenate([x.to_points() for x in np.array(bboxes)[nodes]], axis = 0)
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		merged_box = np.maximum(np.array([
			np.array([min_coord[0], min_coord[1]]),
			np.array([max_coord[0], min_coord[1]]),
			np.array([max_coord[0], max_coord[1]]),
			np.array([min_coord[0], max_coord[1]])
			]), 0)
		# calculate average fg and bg color
		fg_r = round(np.mean([box.fg_r for box in [bboxes[i] for i in nodes]]))
		fg_g = round(np.mean([box.fg_g for box in [bboxes[i] for i in nodes]]))
		fg_b = round(np.mean([box.fg_b for box in [bboxes[i] for i in nodes]]))
		bg_r = round(np.mean([box.bg_r for box in [bboxes[i] for i in nodes]]))
		bg_g = round(np.mean([box.bg_g for box in [bboxes[i] for i in nodes]]))
		bg_b = round(np.mean([box.bg_b for box in [bboxes[i] for i in nodes]]))
		# majority vote for direction
		dirs = [bbox_direction(0, 0, box.w, box.h) for box in [bboxes[i] for i in nodes]]
		majority_dir = Counter(dirs).most_common(1)[0][0]
		# sort
		if majority_dir == 'h' :
			nodes = sorted(nodes, key = lambda x: bboxes[x].y)
		elif majority_dir == 'v' :
			nodes = sorted(nodes, key = lambda x: -(bboxes[x].x + bboxes[x].w))
		# yield overall bbox and sorted indices
		yield merged_box, nodes, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b

def run_detect(model, img_np_resized) :
	img = torch.from_numpy(img_np_resized)
	img = einops.rearrange(img, 'h w c -> 1 c h w')
	with torch.no_grad() :
		craft, mask = model(img)
		rscore = craft[0, 0, :, :].cpu().numpy()
		ascore = craft[0, 1, :, :].cpu().numpy()
		mask = mask[0, 0, :, :].cpu().numpy()
	return rscore, ascore, (mask * 255.0).astype(np.uint8)

def overlay_image(a, b, wa = 0.7) :
	return cv2.addWeighted(a, wa, b, 1 - wa, 0)

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
	with torch.no_grad() :
		return model.infer_beam_batch(img, widths, beams_k = 5, max_seq_length = 255)

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def run_ocr(img, bboxes, dictionary, model, max_chunk_size = 2) :
	text_height = 32
	ret_bboxes = []
	def map_bbox(ubox) :
		x, y, w, h = ubox[0][0], ubox[0][1], ubox[1][0] - ubox[0][0], ubox[2][1] - ubox[1][1]
		real_x, real_y, real_w, real_h = tuple([round(a) for a in [x, y, w, h]])
		char_size = min(w, h)
		x -= char_size * TEXT_EXT_RATIO
		y -= char_size * TEXT_EXT_RATIO
		w += 2 * char_size * TEXT_EXT_RATIO
		h += 2 * char_size * TEXT_EXT_RATIO
		x, y, w, h = tuple([round(a) for a in [x, y, w, h]])
		x = max(x, 0)
		y = max(y, 0)
		w = min(w, img.shape[1] - x - 1)
		h = min(h, img.shape[0] - y - 1)
		region = img[y: y + h, x: x + w]
		if w > h :
			ratio = float(w) / float(h)
			new_width = int(text_height * ratio)
			new_height = text_height
			region = cv2.resize(region, (new_width, new_height), cv2.INTER_LINEAR)
		else :
			ratio = float(h) / float(w)
			new_height = int(text_height * ratio)
			new_width = text_height
			img_resized = cv2.resize(region, (new_width, new_height), cv2.INTER_LINEAR)
			region = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
			new_width, new_height = new_height, new_width
		return (real_x, real_y, real_w, real_h), (new_width, new_height), region
	resized_bboxes = [map_bbox(box) for box in bboxes]
	perm = sorted(range(len(resized_bboxes)), key = lambda x: resized_bboxes[x][1][0])
	for indices in chunks(perm, max_chunk_size) :
		N = len(indices)
		widths = [resized_bboxes[i][1][0] for i in indices]
		max_width = 4 * (max(widths) + 7) // 4
		region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
		for i, idx in enumerate(indices) :
			W = resized_bboxes[idx][1][0]
			region[i, :, : W, :] = resized_bboxes[idx][2]
		images = (torch.from_numpy(region).float() - 127.5) / 127.5
		images = einops.rearrange(images, 'N H W C -> N C H W')
		ret = ocr_infer_bacth(images, model, widths)
		for i, (pred_chars_index, prob, fr, fg, fb, br, bg, bb) in enumerate(ret) :
			if prob < 0.2 :
				continue
			fr = (torch.clip(fr.view(-1), 0, 1).mean() * 255).long().item()
			fg = (torch.clip(fg.view(-1), 0, 1).mean() * 255).long().item()
			fb = (torch.clip(fb.view(-1), 0, 1).mean() * 255).long().item()
			br = (torch.clip(br.view(-1), 0, 1).mean() * 255).long().item()
			bg = (torch.clip(bg.view(-1), 0, 1).mean() * 255).long().item()
			bb = (torch.clip(bb.view(-1), 0, 1).mean() * 255).long().item()
			(x, y, w, h), (new_width, new_height), region = resized_bboxes[indices[i]]
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
			print(prob, txt)
			ret_bboxes.append(BBox(x, y, w, h, txt, prob, fr, fg, fb, br, bg, bb))
	return ret_bboxes

def run_inpainting(model_inpainting, img, mask) :
	img = np.copy(img)
	img[mask > 0] = np.array([255, 255, 255], np.uint8)
	return img

from baidutrans import Translator as baidu_trans
baidu_translator = baidu_trans()

import text_render

def load_ocr_model() :
	with open('alphabet-all-v5.txt', 'r', encoding='utf-8') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	model_ocr = OCR(dictionary, 768)
	model_ocr.load_state_dict(torch.load('ocr.ckpt', map_location='cpu'), strict=False)
	model_ocr.eval()
	return dictionary, model_ocr

def load_detect_model() :
	model = CRAFT_net()
	sd = torch.load('detect.ckpt', map_location='cpu')
	model.load_state_dict(sd['model'])
	model = model.cpu()
	model.eval()
	return model

def load_inpainting_model() :
	return None

def main() :
	print(' -- Loading models')
	import os
	os.makedirs('result', exist_ok = True)
	text_render.prepare_renderer()
	dictionary, model_ocr = load_ocr_model()
	model_detect = load_detect_model()
	model_inpainting = load_inpainting_model()

	print(' -- Read image')
	img = cv2.imread(args.image)
	img_bbox = np.copy(img)
	img_bbox_all = np.copy(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(img, args.size, cv2.INTER_LINEAR, mag_ratio = 1)
	img_to_overlay = np.copy(img_resized)
	ratio_h = ratio_w = 1 / target_ratio
	img_resized = imgproc.normalizeMeanVariance(img_resized)
	print(img_resized.shape)
	print(' -- Running text detection')
	rscore, ascore, mask = run_detect(model_detect, img_resized)
	overlay = imgproc.cvt2HeatmapImg(rscore + ascore)
	boxes, polys = craft_utils.getDetBoxes(rscore, ascore, args.text_threshold, args.link_threshold, args.low_text, False)
	boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h, ratio_net = 2)
	polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2)
	for k in range(len(polys)):
		if polys[k] is None: polys[k] = boxes[k]
	# merge textlines
	polys = merge_bboxes(polys, can_merge_textline)
	for [tl, tr, br, bl] in polys :
		x = int(tl[0])
		y = int(tl[1])
		width = int(tr[0] - tl[0])
		height = int(br[1] - tr[1])
		cv2.rectangle(img_bbox_all, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
	print(' -- Running OCR')
	# run OCR for each textline
	textlines = run_ocr(img_bbox, polys, dictionary, model_ocr, 32)
	# merge textline to text region, filter textlines without characters
	text_regions: List[BBox] = []
	new_textlines = []
	for (poly_regions, textline_indices, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b) in merge_bboxes_text_region(textlines) :
		[tl, tr, br, bl] = poly_regions
		x = int(tl[0]) - 5
		y = int(tl[1]) - 5
		width = int(tr[0] - tl[0]) + 10
		height = int(br[1] - tr[1]) + 10
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
					text += ' ' + textlines[textline_idx].text
			logprob_lengths.append((np.log(textlines[textline_idx].prob), len(textlines[textline_idx].text)))
		vc = count_valuable_text(text)
		total_logprobs = 0.0
		for (logprob, length) in logprob_lengths :
			total_logprobs += logprob * length
		total_logprobs /= sum([x[1] for x in logprob_lengths])
		# filter text region without characters
		if vc > 1 :
			region = BBox(x, y, width, height, text, np.exp(total_logprobs), fg_r, fg_g, fg_b, bg_r, bg_g, bg_b)
			region.textline_indices = []
			region.majority_dir = majority_dir
			text_regions.append(region)
			for textline_idx in textline_indices :
				region.textline_indices.append(len(new_textlines))
				new_textlines.append(textlines[textline_idx])
	textlines = new_textlines
	print(' -- Generating text mask')
	# create mask
	from text_mask_utils import filter_masks, complete_mask
	mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation = cv2.INTER_LINEAR)
	if pad_h > 0 :
		mask_resized = mask_resized[:-pad_h, :]
	elif pad_w > 0 :
		mask_resized = mask_resized[:, : -pad_w]
	mask_resized = cv2.resize(mask_resized, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
	img_resized_2 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation = cv2.INTER_LINEAR)
	mask_resized[mask_resized > 250] = 255
	text_lines = [(a.x // 2, a.y // 2, a.w // 2, a.h // 2) for a in textlines]
	mask_ccs, cc2textline_assignment = filter_masks(mask_resized, text_lines)
	cv2.imwrite('result/mask_filtered.png', reduce(cv2.bitwise_or, mask_ccs))
	final_mask = complete_mask(img_resized_2, mask_ccs, text_lines, cc2textline_assignment)
	final_mask = cv2.resize(final_mask, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_LINEAR)
	print(' -- Running inpainting')
	# run inpainting
	img_inpainted = run_inpainting(model_inpainting, img, final_mask)
	print(' -- Translating')
	# translate text region texts
	texts = '\n'.join([r.text for r in text_regions])
	trans_ret = baidu_translator.translate('ja', 'zh-CN', texts)
	translated_sentences = []
	batch = len(text_regions)
	if len(trans_ret) < batch :
		translated_sentences.extend(trans_ret)
		translated_sentences.extend([''] * (batch - len(trans_ret)))
	elif len(trans_ret) > batch :
		translated_sentences.extend(trans_ret[:batch])
	else :
		translated_sentences.extend(trans_ret)
	print(' -- Rendering translated text')
	# render translated texts
	img_canvas = np.copy(img_inpainted)
	for trans_text, region in zip(translated_sentences, text_regions) :
		print(region.text)
		print(trans_text)
		print(region.majority_dir, region.x, region.y, region.w, region.h)
		img_bbox = cv2.rectangle(img_bbox, (region.x, region.y), (region.x + region.w, region.y + region.h), color=(0, 0, 255), thickness=2)
		fg = (region.fg_b, region.fg_g, region.fg_r)
		for idx in region.textline_indices :
			txtln = textlines[idx]
			img_bbox = cv2.rectangle(img_bbox, (txtln.x, txtln.y), (txtln.x + txtln.w, txtln.y + txtln.h), color = fg, thickness=2)
		if region.majority_dir == 'h' :
			text_render.put_text_horizontal(img_canvas, trans_text, len(region.textline_indices), region.x, region.y, region.w, region.h, fg, None)
		else :
			text_render.put_text_vertical(img_canvas, trans_text, len(region.textline_indices), region.x, region.y, region.w, region.h, fg, None)

	print(' -- Saving results')
	cv2.imwrite('result/rs.png', imgproc.cvt2HeatmapImg(rscore))
	cv2.imwrite('result/as.png', imgproc.cvt2HeatmapImg(ascore))
	cv2.imwrite('result/textline.png', overlay)
	cv2.imwrite('result/bbox.png', img_bbox)
	cv2.imwrite('result/bbox_unfiltered.png', img_bbox_all)
	cv2.imwrite('result/overlay.png', cv2.cvtColor(overlay_image(img_to_overlay, cv2.resize(overlay, (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_LINEAR)), cv2.COLOR_RGB2BGR))
	cv2.imwrite('result/mask.png', final_mask)
	cv2.imwrite('result/masked.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))
	cv2.imwrite('result/final.png', cv2.cvtColor(img_canvas, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
	main()
