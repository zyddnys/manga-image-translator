
import asyncio
import torch
import einops
import argparse
import cv2
import numpy as np
import requests
from oscrypto import util as crypto_utils

parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--mode', default='demo', type=str, help='Run demo in either single image demo mode (demo) or web service mode (web)')
parser.add_argument('--image', default='', type=str, help='Image file if using demo mode')
parser.add_argument('--size', default=1536, type=int, help='image square size')
parser.add_argument('--use-inpainting', action='store_true', help='turn on/off inpainting')
parser.add_argument('--use-cuda', action='store_true', help='turn on/off cuda')
parser.add_argument('--inpainting-size', default=2048, type=int, help='size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='text rendering magnification ratio, larger means higher quality')
parser.add_argument('--translator', default='google', type=str, help='language translator')
parser.add_argument('--target-lang', default='CHS', type=str, help='destination language')
parser.add_argument('--verbose', action='store_true', help='print debug info and save intermediate images')
args = parser.parse_args()
print(args)

def overlay_image(a, b, wa = 0.7) :
	return cv2.addWeighted(a, wa, b, 1 - wa, 0)

def overlay_mask(img, mask) :
	img2 = img.copy().astype(np.float32)
	mask_fp32 = (mask > 10).astype(np.uint8) * 2
	mask_fp32[mask_fp32 == 0] = 1
	mask_fp32 = mask_fp32.astype(np.float32) * 0.5
	img2 = img2 * mask_fp32[:, :, None]
	return img2.astype(np.uint8)

from text_rendering import text_render

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

from detection import dispatch as dispatch_detection, load_model as load_detection_model
from ocr import dispatch as dispatch_ocr, load_model as load_ocr_model
from inpainting import dispatch as dispatch_inpainting, load_model as load_inpainting_model
from text_mask import dispatch as dispatch_mask_refinement
from textline_merge import dispatch as dispatch_textline_merge
from text_rendering import dispatch as dispatch_rendering

async def infer(
	img,
	mode,
	nonce,
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
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'detection')
	textlines, mask = await dispatch_detection(img, img_detect_size, args.use_cuda, args, verbose = args.verbose)

	if args.verbose :
		img_bbox_raw = np.copy(img)
		for txtln in textlines :
			cv2.polylines(img_bbox_raw, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
		cv2.imwrite(f'result/{task_id}/bbox_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'ocr')
	textlines = await dispatch_ocr(img, textlines, args.use_cuda, args)

	text_regions, textlines = await dispatch_textline_merge(textlines, img.shape[1], img.shape[0])
	if args.verbose :
		img_bbox = np.copy(img)
		for region in text_regions :
			for idx in region.textline_indices :
				txtln = textlines[idx]
				cv2.polylines(img_bbox, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
			img_bbox = cv2.polylines(img_bbox, [region.pts], True, color = (0, 0, 255), thickness = 2)
		cv2.imwrite(f'result/{task_id}/bbox.png', cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))

	if mode == 'web' and task_id :
		print(' -- Translating')
		update_state(task_id, nonce, 'translating')
		# in web mode, we can start translation task async
		requests.post('http://127.0.0.1:5003/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.text for r in text_regions]})

	print(' -- Generating text mask')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'mask_generation')
	# create mask
	final_mask = await dispatch_mask_refinement(img, mask, textlines)

	print(' -- Running inpainting')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'inpainting')
	# run inpainting
	img_inpainted = await dispatch_inpainting(args.use_inpainting, False, args.use_cuda, img, final_mask, args.inpainting_size, verbose = args.verbose)
	if args.verbose :
		img_inpainted, inpaint_input = img_inpainted
		cv2.imwrite(f'result/{task_id}/inpaint_input.png', cv2.cvtColor(inpaint_input, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/mask_final.png', final_mask)

	# translate text region texts
	if mode != 'web' :
		print(' -- Translating')
		from translators import dispatch as run_translation
		translated_sentences = await run_translation(args.translator, 'auto', args.target_lang, [r.text for r in text_regions])
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
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'render')
	# render translated texts
	output = await dispatch_rendering(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, textlines, text_regions)
	
	print(' -- Saving results')
	cv2.imwrite(f'result/{task_id}/final.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

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
	with open('alphabet-all-v5.txt', 'r', encoding = 'utf-8') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	load_ocr_model(dictionary, args.use_cuda)
	load_detection_model(args.use_cuda)
	load_inpainting_model(args.use_cuda)

	if mode == 'demo' :
		print(' -- Running in single image demo mode')
		if not args.image :
			print('please provide an image')
			parser.print_usage()
			return
		img = cv2.imread(args.image)
		await infer(img, mode, '')
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
					infer_task = asyncio.create_task(infer(img, mode, nonce, task_id))
					asyncio.gather(infer_task)
				except :
					import traceback
					traceback.print_exc()
					update_state(task_id, nonce, 'error')
			else :
				await asyncio.sleep(0.1)
	

if __name__ == '__main__':
	asyncio.run(main(args.mode))
