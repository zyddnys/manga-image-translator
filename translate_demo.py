import asyncio
import argparse
import time
from PIL import Image
import cv2
import numpy as np
import requests
import os
from oscrypto import util as crypto_utils
import asyncio
import torch

from detection import DETECTORS, dispatch as dispatch_detection, prepare as prepare_detection
from detection.ctd_utils.textblock import visualize_textblocks
from ocr import OCRS, dispatch as dispatch_ocr, prepare as prepare_ocr
from inpainting import INPAINTERS, dispatch as dispatch_inpainting, prepare as prepare_inpainting
from translators import OFFLINE_TRANSLATORS, TRANSLATORS, VALID_LANGUAGES, dispatch as dispatch_translation, prepare as prepare_translation
from upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from text_rendering import text_render, dispatch_ctd_render
from utils import load_image, dump_image

parser = argparse.ArgumentParser(description='Seamlessly translate mangas into a chosen language')
parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'batch', 'web', 'web2'], help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('-i', '--image', default='', type=str, help='Path to an image file if using demo mode, or path to an image folder if using batch mode')
parser.add_argument('-o', '--image-dst', default='', type=str, help='Path to the destination folder for translated images in batch mode')
parser.add_argument('-l', '--target-lang', default='CHS', type=str, choices=VALID_LANGUAGES, help='Destination language')
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--detector', default='default', type=str, choices=DETECTORS, help='Text detector used for creating a text mask from an image')
parser.add_argument('--ocr', default='48px_ctc', type=str, choices=OCRS, help='Optical character recognition (OCR) model to use')
parser.add_argument('--inpainter', default='lama_mpe', type=str, choices=INPAINTERS, help='Inpainting model to use')
parser.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='Language translator to use')
parser.add_argument('--mtpe', action='store_true', help='Turn on/off machine translation post editing (MTPE) on the command line')
parser.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
parser.add_argument('--use-cuda-limited', action='store_true', help='Turn on/off cuda (excluding offline translator)')
parser.add_argument('--detection-size', default=1536, type=int, help='Size of image used for detection')
parser.add_argument('--det-rearrange-max-batches', default=4, type=int, help='Max batch size produced by the rearrangement of image with extreme aspectio, reduce it if cuda OOM')
parser.add_argument('--inpainting-size', default=2048, type=int, help='Size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='Threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='Threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='Text rendering magnification ratio, larger means higher quality')
parser.add_argument('--font-size-offset', default=0, type=int, help='Offset font size by a given amount, positive number increase font size and vice versa')
parser.add_argument('--force-horizontal', action='store_true', help='Force text to be rendered horizontally')
parser.add_argument('--force-vertical', action='store_true', help='Force text to be rendered vertically')
parser.add_argument('--upscale-ratio', default=None, type=int, choices=[1, 2, 4, 8, 16, 32], help='waifu2x image upscale ratio')
parser.add_argument('--manga2eng', action='store_true', help='Render english text translated from manga with some typesetting')
parser.add_argument('--eng-font', default='fonts/comic shanns 2.ttf', type=str, help='Path to font used by manga2eng mode')
args = parser.parse_args()

def update_state(task_id, nonce, state):
	while True:
		try:
			requests.post(f'http://{args.host}:{args.port}/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state}, timeout = 20)
			return
		except Exception:
			if 'error' in state or 'finished' in state:
				continue
			else:
				break

def get_task(nonce):
	try:
		rjson = requests.get(f'http://{args.host}:{args.port}/task-internal?nonce={nonce}', timeout = 3600).json()
		if 'task_id' in rjson and 'data' in rjson:
			return rjson['task_id'], rjson['data']
		elif 'data' in rjson:
			return None, rjson['data']
		return None, None
	except Exception:
		return None, None

async def infer(
	image: Image.Image,
	mode,
	nonce = '',
	options = None,
	task_id = '',
	dst_image_name = '',
	):

	img_rgb, img_alpha = load_image(image)

	options = options or {}
	img_detect_size = args.detection_size
	if 'size' in options:
		size_ind = options['size']
		if size_ind == 'S':
			img_detect_size = 1024
		elif size_ind == 'M':
			img_detect_size = 1536
		elif size_ind == 'L':
			img_detect_size = 2048
		elif size_ind == 'X':
			img_detect_size = 2560

	if 'detector' in options:
		detector = options['detector']
	else:
		detector = args.detector

	render_text_direction_overwrite = options.get('direction')
	if not render_text_direction_overwrite:
		if args.force_horizontal:
			render_text_direction_overwrite = 'h'
		elif args.force_vertical:
			render_text_direction_overwrite = 'v'
		else:
			render_text_direction_overwrite = 'auto'

	src_lang = 'auto'
	if 'tgt' in options:
		tgt_lang = options['tgt']
	else:
		tgt_lang = args.target_lang
	if 'translator' in options:
		translator = options['translator']
	else:
		translator = args.translator

	if not dst_image_name:
		dst_image_name = f'result/{task_id}/final.png'

	print(f' -- Detection resolution {img_detect_size}')
	print(f' -- Detector using {detector}')
	print(f' -- Render text direction is {render_text_direction_overwrite}')

	print(' -- Preparing translator')
	await prepare_translation(translator, src_lang, tgt_lang)

	# The default text detector doesn't work very well on smaller images, so small images
	# will get upscaled automatically unless --upscale-ratio=1 was set
	if args.upscale_ratio or image.size[0] < 800 or image.size[1] < 800:
		print(' -- Running upscaling')
		if mode == 'web' and task_id:
			update_state(task_id, nonce, 'upscaling')

		if args.upscale_ratio:
			ratio = args.upscale_ratio
		else:
			ratio = max(4, 800 / image.size[0], 800 / image.size[1])
		img_upscaled_pil = (await dispatch_upscaling('waifu2x', [image], ratio, args.use_cuda))[0]
		img_rgb, img_alpha = load_image(img_upscaled_pil)

	print(' -- Running text detection')
	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'detection')

	textlines, final_mask = await dispatch_detection(args.detector, img_rgb, img_detect_size, args.text_threshold, args.box_threshold, args.unclip_ratio, args.det_rearrange_max_batches, args.verbose, args.use_cuda)
	text_regions = textlines

	if args.verbose:
		cv2.imwrite(f'result/{task_id}/mask_final.png', final_mask)
		bboxes = visualize_textblocks(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB), textlines)
		cv2.imwrite(f'result/{task_id}/bboxes.png', bboxes)

	print(' -- Running OCR')
	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'ocr')
	textlines = await dispatch_ocr(args.ocr, img_rgb, textlines, args.use_cuda, args.verbose)

	# in web mode, we can start online translation tasks async
	if mode == 'web' and task_id and options.get('translator') not in OFFLINE_TRANSLATORS:
		update_state(task_id, nonce, 'translating')
		requests.post(f'http://{args.host}:{args.port}/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [r.get_text() for r in text_regions]}, timeout = 20)

	if args.verbose:
		inpaint_input_img = await dispatch_inpainting('none', img_rgb, final_mask)
		cv2.imwrite(f'result/{task_id}/inpaint_input.png', cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))

	if text_regions:
		print(' -- Running inpainting')
		if mode == 'web' and task_id:
			update_state(task_id, nonce, 'inpainting')
		img_inpainted = await dispatch_inpainting(args.inpainter, img_rgb, final_mask, args.inpainting_size, args.verbose, args.use_cuda)
	else:
		img_inpainted = img_rgb

	if args.verbose:
		cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))

	print(' -- Translating')
	translated_sentences = None
	if mode != 'web' or translator in OFFLINE_TRANSLATORS:
		if mode == 'web' and task_id:
			update_state(task_id, nonce, 'translating')
		queries = [r.get_text() for r in text_regions]
		translated_sentences = await dispatch_translation(translator, src_lang, tgt_lang, queries, args.mtpe, use_cuda=args.use_cuda and not args.use_cuda_limited)
	else:
		# wait for at most 1 hour for manual translation
		if options.get('manual', False):
			wait_for = 3600
		else:
			wait_for = 30 # 30 seconds for machine translation
		wait_until = time.time() + wait_for
		while time.time() < wait_until:
			ret = requests.post(f'http://{args.host}:{args.port}/get-translation-result-internal', json = {'task_id': task_id, 'nonce': nonce}, timeout = 20).json()
			if 'result' in ret:
				translated_sentences = ret['result']
				if isinstance(translated_sentences, str):
					if translated_sentences == 'error':
						update_state(task_id, nonce, 'error-lang')
						return
				break
			await asyncio.sleep(0.01)

	print(' -- Rendering translated text')
	if translated_sentences == None:
		print("No text found!")
		if mode == 'web' and task_id:
			update_state(task_id, nonce, 'error-no-txt')
		return

	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'render')
	# render translated texts
	if tgt_lang == 'ENG' and args.manga2eng:
		from text_rendering import dispatch_eng_render
		output = await dispatch_eng_render(np.copy(img_inpainted), img_rgb, text_regions, translated_sentences, args.eng_font)
	else:
		output = await dispatch_ctd_render(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, text_regions, render_text_direction_overwrite, tgt_lang, args.font_size_offset)

	print(' -- Saving results')
	img_pil = dump_image(output, img_alpha)
	img_pil.save(dst_image_name)

	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'finished')


async def infer_safe(
	img: Image.Image,
	mode,
	nonce,
	options = None,
	task_id = '',
	dst_image_name = '',
	):
	try:
		return await infer(
			img,
			mode,
			nonce,
			options,
			task_id,
			dst_image_name,
		)
	except Exception:
		import traceback
		traceback.print_exc()
		update_state(task_id, nonce, 'error')

def replace_prefix(s: str, old: str, new: str):
	if s.startswith(old):
		s = new + s[len(old):]
	return s

async def main(mode = 'demo'):
	print(' -- Preload Checks')
	args.image = os.path.expanduser(args.image)
	if not os.path.exists(args.image):
		raise FileNotFoundError(args.image)

	if args.use_cuda_limited:
		args.use_cuda = True
	if not torch.cuda.is_available() and args.use_cuda:
		raise Exception('CUDA compatible device could not be found while %s args was set...'
						% ('--use_cuda_limited' if args.use_cuda_limited else '--use_cuda'))

	print(' -- Loading models')
	os.makedirs('result', exist_ok=True)
	text_render.prepare_renderer()
	await prepare_upscaling('waifu2x')
	await prepare_detection(args.detector)
	await prepare_ocr(args.ocr, args.use_cuda)
	await prepare_inpainting(args.inpainter, args.use_cuda)

	if mode == 'demo':
		print(' -- Running in single image demo mode')
		if not args.image:
			print('please provide an image')
			parser.print_usage()
			return
		await infer(Image.open(args.image), mode)
	elif mode == 'web' or mode == 'web2':
		print(' -- Running in web service mode')
		print(' -- Waiting for translation tasks')

		if mode == 'web':
			import subprocess
			import sys
			nonce = crypto_utils.rand_bytes(16).hex()

			extra_web_args = {'stdout':sys.stdout, 'stderr':sys.stderr} if args.log_web else {}
			web_executable = [sys.executable, '-u'] if args.log_web else [sys.executable]
			web_process_args = ['web_main.py', nonce, str(args.host), str(args.port)]
			subprocess.Popen([*web_executable, *web_process_args], **extra_web_args)

		while True:
			try:
				task_id, options = get_task(nonce)
				if options and 'exit' in options:
					break
				if task_id:
					try:
						print(f' -- Processing task {task_id}')
						infer_task = asyncio.create_task(infer_safe(Image.open(f'result/{task_id}/input.png'), mode, nonce, options, task_id))
						asyncio.gather(infer_task)
					except Exception:
						import traceback
						traceback.print_exc()
						update_state(task_id, nonce, 'error')
				else:
					await asyncio.sleep(0.1)
			except Exception:
				import traceback
				traceback.print_exc()
	elif mode == 'batch':
		src = os.path.abspath(args.image)
		if src[-1] == '\\' or src[-1] == '/':
			src = src[:-1]
		dst = args.image_dst or src + '-translated'
		if os.path.exists(dst) and not os.path.isdir(dst):
			print(f'Destination `{dst}` already exists and is not a directory! Please specify another directory.')
			return
		print('Processing image in source directory')
		files = []
		for root, subdirs, files in os.walk(src):
			dst_root = replace_prefix(root, src, dst)
			os.makedirs(dst_root, exist_ok = True)
			for f in files:
				if f.lower() == '.thumb':
					continue
				filename = os.path.join(root, f)
				dst_filename = replace_prefix(filename, src, dst)
				if os.path.exists(dst_filename):
					continue
				try:
					img = Image.open(filename)
				except Exception:
					pass
				try:
					print('Processing', filename, '->', dst_filename)
					await infer(img, 'demo', dst_image_name = dst_filename)
				except Exception:
					import traceback
					traceback.print_exc()
					pass

if __name__ == '__main__':
	try:
		print(args)
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		loop.run_until_complete(main(args.mode))
	except KeyboardInterrupt:
		print()
