
import asyncio
import argparse
from PIL import Image
import cv2
import numpy as np
import requests
import os
from oscrypto import util as crypto_utils
import asyncio

from detection import dispatch as dispatch_detection, load_model as load_detection_model
from ocr import dispatch as dispatch_ocr, load_model as load_ocr_model
from inpainting import dispatch as dispatch_inpainting, load_model as load_inpainting_model
from text_mask import dispatch as dispatch_mask_refinement
from textline_merge import dispatch as dispatch_textline_merge
from text_rendering import dispatch as dispatch_rendering, text_render
from textblockdetector import dispatch as dispatch_ctd_detection
from textblockdetector.textblock import visualize_textblocks
from utils import convert_img

parser = argparse.ArgumentParser(description='Generate text bboxes given a image file')
parser.add_argument('--mode', default='demo', type=str, help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('--image', default='', type=str, help='Image file if using demo mode or Image folder name if using batch mode')
parser.add_argument('--image-dst', default='', type=str, help='Destination folder for translated images in batch mode')
parser.add_argument('--size', default=1536, type=int, help='image square size')
parser.add_argument('--use-inpainting', action='store_true', help='turn on/off inpainting')
parser.add_argument('--use-cuda', action='store_true', help='turn on/off cuda')
parser.add_argument('--force-horizontal', action='store_true', help='force texts rendered horizontally')
parser.add_argument('--inpainting-size', default=2048, type=int, help='size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='text rendering magnification ratio, larger means higher quality')
parser.add_argument('--translator', default='google', type=str, help='language translator')
parser.add_argument('--target-lang', default='CHS', type=str, help='destination language')
parser.add_argument('--use-ctd', action='store_true', help='use comic-text-detector for text detection')
parser.add_argument('--verbose', action='store_true', help='print debug info and save intermediate images')
args = parser.parse_args()

def update_state(task_id, nonce, state) :
	requests.post('http://127.0.0.1:5003/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state})

def get_task(nonce) :
	try :
		rjson = requests.get(f'http://127.0.0.1:5003/task-internal?nonce={nonce}').json()
		if 'task_id' in rjson and 'data' in rjson :
			return rjson['task_id'], rjson['data']
		else :
			return None, None
	except :
		return None, None

async def infer(
	img,
	mode,
	nonce,
	options = None,
	task_id = '',
	dst_image_name = '',
	alpha_ch = None
	) :
	options = options or {}
	img_detect_size = args.size
	if 'size' in options :
		size_ind = options['size']
		if size_ind == 'S' :
			img_detect_size = 1024
		elif size_ind == 'M' :
			img_detect_size = 1536
		elif size_ind == 'L' :
			img_detect_size = 2048
		elif size_ind == 'X' :
			img_detect_size = 2560
	print(f' -- Detection resolution {img_detect_size}')
	detector = 'ctd' if args.use_ctd else 'default'
	if 'detector' in options :
		detector = options['detector']
	print(f' -- Detector using {detector}')
	render_text_direction_overwrite = 'h' if args.force_horizontal else ''
	if 'direction' in options :
		if options['direction'] == 'horizontal' :
			render_text_direction_overwrite = 'h'
	print(f' -- Render text direction is {render_text_direction_overwrite or "auto"}')

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'detection')
	
	if detector == 'ctd' :
		mask, final_mask, textlines = await dispatch_ctd_detection(img, args.use_cuda)
		text_regions = textlines
	else:
		textlines, mask = await dispatch_detection(img, img_detect_size, args.use_cuda, args, verbose = args.verbose)

	if args.verbose :
		if detector == 'ctd' :
			bboxes = visualize_textblocks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), textlines)
			cv2.imwrite(f'result/{task_id}/bboxes.png', bboxes)
			cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)
			cv2.imwrite(f'result/{task_id}/mask_final.png', final_mask)
		else:
			img_bbox_raw = np.copy(img)
			for txtln in textlines :
				cv2.polylines(img_bbox_raw, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
			cv2.imwrite(f'result/{task_id}/bbox_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))
			cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'ocr')
	textlines = await dispatch_ocr(img, textlines, args.use_cuda, args)

	if detector == 'default' :
		text_regions, textlines = await dispatch_textline_merge(textlines, img.shape[1], img.shape[0], verbose = args.verbose)
		if args.verbose :
			img_bbox = np.copy(img)
			for region in text_regions :
				for idx in region.textline_indices :
					txtln = textlines[idx]
					cv2.polylines(img_bbox, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
				img_bbox = cv2.polylines(img_bbox, [region.pts], True, color = (0, 0, 255), thickness = 2)
			cv2.imwrite(f'result/{task_id}/bbox.png', cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))

		print(' -- Generating text mask')
		if mode == 'web' and task_id :
			update_state(task_id, nonce, 'mask_generation')
		# create mask
		final_mask = await dispatch_mask_refinement(img, mask, textlines)

	if mode == 'web' and task_id :
		print(' -- Translating')
		update_state(task_id, nonce, 'translating')
		# in web mode, we can start translation task async
		if detector == 'ctd':
			requests.post('http://127.0.0.1:5003/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.get_text() for r in text_regions]})
		else:
			requests.post('http://127.0.0.1:5003/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.text for r in text_regions]})

	print(' -- Running inpainting')
	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'inpainting')
	# run inpainting
	if text_regions :
		img_inpainted = await dispatch_inpainting(args.use_inpainting, False, args.use_cuda, img, final_mask, args.inpainting_size, verbose = args.verbose)
	else :
		img_inpainted = img, img
	if args.verbose :
		img_inpainted, inpaint_input = img_inpainted
		cv2.imwrite(f'result/{task_id}/inpaint_input.png', cv2.cvtColor(inpaint_input, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))
		cv2.imwrite(f'result/{task_id}/mask_final.png', final_mask)

	# translate text region texts
	translated_sentences = None
	if mode != 'web' :
		print(' -- Translating')
		# try:
		from translators import dispatch as run_translation
		if detector == 'ctd' :
			translated_sentences = await run_translation(args.translator, 'auto', args.target_lang, [r.get_text() for r in text_regions])
		else:
			translated_sentences = await run_translation(args.translator, 'auto', args.target_lang, [r.text for r in text_regions])

	else :
		# wait for at most 1 hour
		for _ in range(36000) :
			ret = requests.post('http://127.0.0.1:5003/get-translation-result-internal', json = {'task_id': task_id, 'nonce': nonce}).json()
			if 'result' in ret :
				translated_sentences = ret['result']
				break
			await asyncio.sleep(0.1)

	if translated_sentences is not None:
		print(' -- Rendering translated text')
		if mode == 'web' and task_id :
			update_state(task_id, nonce, 'render')
		# render translated texts
		if detector == 'ctd' :
			from text_rendering import dispatch_ctd_render
			output = await dispatch_ctd_render(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, text_regions, render_text_direction_overwrite)
		else:
			output = await dispatch_rendering(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, textlines, text_regions, render_text_direction_overwrite)
		
		print(' -- Saving results')
		if alpha_ch is not None :
			output = np.concatenate([output, np.array(alpha_ch)[..., None]], axis = 2)
		img_pil = Image.fromarray(output)
		if dst_image_name :
			img_pil.save(dst_image_name)
		else :
			img_pil.save(f'result/{task_id}/final.png')

	if mode == 'web' and task_id :
		update_state(task_id, nonce, 'finished')

def replace_prefix(s: str, old: str, new: str) :
	if s.startswith(old) :
		s = new + s[len(old):]
	return s

async def main(mode = 'demo') :
	print(' -- Loading models')
	os.makedirs('result', exist_ok = True)
	text_render.prepare_renderer()
	with open('alphabet-all-v5.txt', 'r', encoding = 'utf-8') as fp :
		dictionary = [s[:-1] for s in fp.readlines()]
	load_ocr_model(dictionary, args.use_cuda)
	from textblockdetector import load_model as load_ctd_model
	load_ctd_model(args.use_cuda)
	load_detection_model(args.use_cuda)
	load_inpainting_model(args.use_cuda)

	if mode == 'demo' :
		print(' -- Running in single image demo mode')
		if not args.image :
			print('please provide an image')
			parser.print_usage()
			return
		img, alpha_ch = convert_img(Image.open(args.image))
		img = np.array(img)
		await infer(img, mode, '', alpha_ch = alpha_ch)
	elif mode == 'web' :
		print(' -- Running in web service mode')
		print(' -- Waiting for translation tasks')
		nonce = crypto_utils.rand_bytes(16).hex()
		import subprocess
		import sys
		subprocess.Popen([sys.executable, 'web_main.py', nonce, '5003'])
		while True :
			task_id, options = get_task(nonce)
			if task_id :
				print(f' -- Processing task {task_id}')
				img, alpha_ch = convert_img(Image.open(args.image))
				img = np.array(img)
				try :
					infer_task = asyncio.create_task(infer(img, mode, nonce, options, task_id, alpha_ch = alpha_ch))
					asyncio.gather(infer_task)
				except :
					import traceback
					traceback.print_exc()
					update_state(task_id, nonce, 'error')
			else :
				await asyncio.sleep(0.1)
	elif mode == 'batch' :
		src = os.path.abspath(args.image)
		if src[-1] == '\\' or src[-1] == '/' :
			src = src[:-1]
		dst = args.image_dst or src + '-translated'
		if os.path.exists(dst) and not os.path.isdir(dst) :
			print(f'Destination `{dst}` already exists and is not a directory! Please specify another directory.')
			return
		if os.path.exists(dst) and os.listdir(dst) :
			print(f'Destination directory `{dst}` already exists! Please specify another directory.')
			return
		print('Processing image in source directory')
		files = []
		for root, subdirs, files in os.walk(src) :
			dst_root = replace_prefix(root, src, dst)
			os.makedirs(dst_root, exist_ok = True)
			for f in files :
				if f.lower() == '.thumb' :
					continue
				filename = os.path.join(root, f)
				try :
					img, alpha_ch = convert_img(Image.open(filename))
					img = np.array(img)
					if img is None :
						continue
				except Exception :
					pass
				try :
					dst_filename = replace_prefix(filename, src, dst)
					print('Processing', filename, '->', dst_filename)
					await infer(img, 'demo', '', dst_image_name = dst_filename, alpha_ch = alpha_ch)
				except Exception :
					import traceback
					traceback.print_exc()
					pass

if __name__ == '__main__':
	print(args)
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main(args.mode))
