
import torch
import cv2
import numpy as np
from .inpainting_aot import AOTGenerator
from .inpainting_lama import get_generator as get_lama_generator
from .inpainting_lama_mpe import load_lama_mpe, LamaFourier
from utils import resize_keep_aspect

INPAINTING_MODEL = None

def load_model(cuda: bool, model_name: str = 'default') :
	global INPAINTING_MODEL
	if model_name not in ['default', 'lama', 'lama_mpe'] :
		raise Exception
	if model_name == 'default' and INPAINTING_MODEL is None :
		model = AOTGenerator()
		sd = torch.load('inpainting.ckpt', map_location = 'cpu')
		model.load_state_dict(sd['model'] if 'model' in sd else sd)
		model.eval()
		if cuda :
			model = model.cuda()
		INPAINTING_MODEL = model
	if model_name == 'lama' and INPAINTING_MODEL is None :
		model = get_lama_generator()
		sd = torch.load('inpainting_lama.ckpt', map_location = 'cpu')
		model.load_state_dict(sd['model'] if 'model' in sd else sd)
		model.eval()
		if cuda :
			model = model.cuda()
		INPAINTING_MODEL = model
	if model_name == 'lama_mpe' and INPAINTING_MODEL is None :
		model = load_lama_mpe('inpainting_lama_mpe.ckpt', device='cpu')
		model.eval()
		if cuda :
			model = model.cuda()
		INPAINTING_MODEL = model

async def dispatch(use_inpainting: bool, use_poisson_blending: bool, cuda: bool, img: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, model_name: str = 'default', verbose: bool = False) -> np.ndarray :

	img_original = np.copy(img)
	mask_original = np.copy(mask)
	mask_original[mask_original < 127] = 0
	mask_original[mask_original >= 127] = 1
	mask_original = mask_original[:, :, None]
	if not use_inpainting :
		img = np.copy(img)
		img[mask > 0] = np.array([255, 255, 255], np.uint8)
		return img
	height, width, c = img.shape
	if max(img.shape[0: 2]) > inpainting_size :
		img = resize_keep_aspect(img, inpainting_size)
		mask = resize_keep_aspect(mask, inpainting_size)
	pad_size = 8
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
		img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
		mask = cv2.resize(mask, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
	if verbose :
		print(f'Inpainting resolution: {new_w}x{new_h}')
	if isinstance(INPAINTING_MODEL, LamaFourier):
		img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.
	else:
		img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
	mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
	mask_torch[mask_torch < 0.5] = 0
	mask_torch[mask_torch >= 0.5] = 1
	if cuda :
		img_torch = img_torch.cuda()
		mask_torch = mask_torch.cuda()
	with torch.no_grad() :
		img_torch *= (1 - mask_torch)
		img_inpainted_torch = INPAINTING_MODEL(img_torch, mask_torch)
	if isinstance(INPAINTING_MODEL, LamaFourier):
		img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
	else:
		img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
	if new_h != height or new_w != width :
		img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation = cv2.INTER_LINEAR)
	if use_poisson_blending :
		raise NotImplemented
	else :
		ans = img_inpainted * mask_original + img_original * (1 - mask_original)
	return ans
