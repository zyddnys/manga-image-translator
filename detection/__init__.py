
import torch
import cv2
from typing import List
import numpy as np
from utils import Quadrilateral
from .DBNet_resnet34 import TextDetection as TextDetectionDefault
from . import imgproc, dbnet_utils, craft_utils
import einops

DEFAULT_MODEL = None

def load_model(cuda: bool, model_name: str = 'default') :
	global DEFAULT_MODEL
	if model_name not in ['default'] :
		raise Exception
	if model_name == 'default' and DEFAULT_MODEL is None :
		model = TextDetectionDefault()
		sd = torch.load('detect.ckpt', map_location = 'cpu')
		model.load_state_dict(sd['model'] if 'model' in sd else sd)
		model.eval()
		if cuda :
			model = model.cuda()
		DEFAULT_MODEL = model

async def run_default(img: np.ndarray, detect_size: int, cuda: bool, verbose: bool, args: dict) :
	global DEFAULT_MODEL
	img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(cv2.bilateralFilter(img, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
	ratio_h = ratio_w = 1 / target_ratio
	if verbose :
		print(f'Detection resolution: {img_resized.shape[1]}x{img_resized.shape[0]}')
	img_resized = img_resized.astype(np.float32) / 127.5 - 1.0
	img = torch.from_numpy(img_resized)
	if cuda :
		img = img.cuda()
	img = einops.rearrange(img, 'h w c -> 1 c h w')
	with torch.no_grad() :
		db, mask = DEFAULT_MODEL(img)
		db = db.sigmoid().cpu()
		mask = mask[0, 0, :, :].cpu().numpy()
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
	textlines = list(filter(lambda q: q.area > 16, textlines))
	mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation = cv2.INTER_LINEAR)
	if pad_h > 0 :
		mask_resized = mask_resized[:-pad_h, :]
	elif pad_w > 0 :
		mask_resized = mask_resized[:, : -pad_w]
	return textlines, np.clip(mask_resized * 255, 0, 255).astype(np.uint8)

async def dispatch(img: np.ndarray, detect_size: int, cuda: bool, args: dict, model_name: str = 'default', verbose: bool = False) -> List[Quadrilateral] :
	print(' -- Running text detection')
	if model_name == 'default' :
		global DEFAULT_MODEL
		if DEFAULT_MODEL is None :
			load_model(cuda, 'default')
		return await run_default(img, detect_size, cuda, verbose, args)

