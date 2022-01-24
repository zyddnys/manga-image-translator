
from collections import Counter
import itertools
from typing import List, Tuple, Union
from utils import Quadrilateral, quadrilateral_can_merge_region
import torch
import cv2
import numpy as np
import einops
import networkx as nx

from .model_32px import OCR as OCR_32px
from .model_48px import OCR as OCR_48px

from textblockdetector.textblock import TextBlock

MODEL_32PX = None

def load_model(dictionary, cuda: bool, model_name: str = '32px') :
	global MODEL_32PX
	if model_name not in ['32px'] :
		raise Exception
	if model_name == '32px' and MODEL_32PX is None :
		model = OCR_32px(dictionary, 768)
		sd = torch.load('ocr.ckpt', map_location = 'cpu')
		model.load_state_dict(sd['model'] if 'model' in sd else sd)
		model.eval()
		if cuda :
			model = model.cuda()
		MODEL_32PX = model

def ocr_infer_bacth(img, model, widths) :
	with torch.no_grad() :
		return model.infer_beam_batch(img, widths, beams_k = 5, max_seq_length = 255)

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

def run_ocr_32px(img: np.ndarray, cuda: bool, quadrilaterals: List[Tuple[Union[Quadrilateral, TextBlock], str]], max_chunk_size = 16, verbose: bool = False) :
	text_height = 32
	regions = [q.get_transformed_region(img, d, text_height) for q, d in quadrilaterals]
	out_regions = []

	perm = range(len(regions))
	if len(quadrilaterals) > 0: 
		if isinstance(quadrilaterals[0][0], Quadrilateral):
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
			if verbose :
				if quadrilaterals[idx][1] == 'v' :
					cv2.imwrite(f'ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
				else :
					cv2.imwrite(f'ocrs/{ix}.png', cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
			ix += 1
		images = (torch.from_numpy(region).float() - 127.5) / 127.5
		images = einops.rearrange(images, 'N H W C -> N C H W')
		if cuda :
			images = images.cuda()
		ret = ocr_infer_bacth(images, MODEL_32PX, widths)
		for i, (pred_chars_index, prob, fr, fg, fb, br, bg, bb) in enumerate(ret) :
			if prob < 0.8 :
				continue
			fr = (torch.clip(fr.view(-1), 0, 1).mean() * 255).long().item()
			fg = (torch.clip(fg.view(-1), 0, 1).mean() * 255).long().item()
			fb = (torch.clip(fb.view(-1), 0, 1).mean() * 255).long().item()
			br = (torch.clip(br.view(-1), 0, 1).mean() * 255).long().item()
			bg = (torch.clip(bg.view(-1), 0, 1).mean() * 255).long().item()
			bb = (torch.clip(bb.view(-1), 0, 1).mean() * 255).long().item()
			seq = []
			for chid in pred_chars_index :
				ch = MODEL_32PX.dictionary[chid]
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
			if isinstance(cur_region, Quadrilateral):
				cur_region.text = txt
				cur_region.prob = prob
				cur_region.fg_r = fr
				cur_region.fg_g = fg
				cur_region.fg_b = fb
				cur_region.bg_r = br
				cur_region.bg_g = bg
				cur_region.bg_b = bb
			else:
				cur_region.text.append(txt)
				cur_region.fg_r += fr
				cur_region.fg_g += fg
				cur_region.fg_b += fb
				cur_region.bg_r += br
				cur_region.bg_g += bg
				cur_region.bg_b += bb

			out_regions.append(cur_region)
	return out_regions

def generate_text_direction(bboxes: List[Union[Quadrilateral, TextBlock]]) :
	if len(bboxes) > 0:
		if isinstance(bboxes[0], TextBlock):
			for blk in bboxes:
				majority_dir = 'v' if blk.vertical else 'h'
				for line_idx in range(len(blk.lines)):
					yield blk, line_idx
		else:
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

async def dispatch(img: np.ndarray, textlines: List[Union[Quadrilateral, TextBlock]], cuda: bool, args: dict, model_name: str = '32px', batch_size: int = 16, verbose: bool = False) -> List[Quadrilateral] :
	print(' -- Running OCR')
	if model_name == '32px' :
		return run_ocr_32px(img, cuda, list(generate_text_direction(textlines)), batch_size)
