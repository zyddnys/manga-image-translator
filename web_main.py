import io
import os
import sys
import time
import asyncio
import traceback
import PIL
import copy
from PIL import Image
from oscrypto import util as crypto_utils
from aiohttp import web
from aiohttp import ClientSession

from collections import deque

from translators import VALID_LANGUAGES, dispatch as run_translation

NONCE = ''
QUEUE = deque()
TASK_DATA = {}
TASK_STATES = {}

app = web.Application(client_max_size = 1024 * 1024 * 10)
routes = web.RouteTableDef()

def convert_img(img) :
	if img.mode == 'RGBA' :
		# from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		background.paste(img, mask = img.split()[3])  # 3 is the alpha channel
		return background
	elif img.mode == 'P' :
		img = img.convert('RGBA')
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		background.paste(img, mask = img.split()[3])  # 3 is the alpha channel
		return background
	else :
		return img.convert('RGB')


@routes.get("/")
async def index_async(request):
	with open('ui.html', 'r') as fp :
		return web.Response(text=fp.read(), content_type='text/html')

@routes.post("/run")
async def run_async(request):
	data = await request.post()
	size = ''
	selected_translator = 'youdao'
	target_language = 'CHS'
	if 'tgt_lang' in data :
		target_language = data['tgt_lang'].upper()
		if target_language not in VALID_LANGUAGES :
			target_language = 'CHS'
	if 'translator' in data :
		selected_translator = data['translator'].lower()
		if selected_translator not in ['youdao', 'baidu'] :
			selected_translator = 'youdao'
	if 'size' in data :
		size = data['size'].upper()
		if size not in ['S', 'M', 'L', 'X'] :
			size = ''
	if 'file' in data :
		file_field = data['file']
		content = file_field.file.read()
	elif 'url' in data :
		from aiohttp import ClientSession
		async with ClientSession() as session:
			async with session.get(data['url']) as resp:
				if resp.status == 200 :
					content = await resp.read()
				else :
					return web.json_response({'status' : 'failed'})
	else :
		return web.json_response({'status' : 'failed'})
	try :
		img = convert_img(Image.open(io.BytesIO(content)))
	except :
		return web.json_response({'status' : 'failed'})
	task_id = crypto_utils.rand_bytes(16).hex() + size
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'translator': selected_translator, 'tgt': target_language}
	TASK_STATES[task_id] = 'pending'
	while True :
		await asyncio.sleep(0.05)
		state = TASK_STATES[task_id]
		if state == 'finished' :
			break
	return web.json_response({'task_id' : task_id, 'status': 'successful'})


@routes.get("/task-internal")
async def get_task_async(request):
	global NONCE
	if request.rel_url.query['nonce'] == NONCE :
		if len(QUEUE) > 0 :
			item = QUEUE.popleft()
			return web.json_response({'task_id': item})
		else :
			return web.json_response({})
	else :
		print('unauthorized', request.rel_url.query['nonce'], NONCE)
	return web.json_response({})

async def machine_trans_task(task_id, texts, translator = 'youdao', target_language = 'CHS') :
	print('translator', translator)
	print('target_language', target_language)
	if texts :
		try :
			TASK_DATA[task_id]['trans_result'] = await run_translation(translator, 'auto', target_language, texts)
		except Exception as ex :
			TASK_DATA[task_id]['trans_result'] = ['error'] * len(texts)
	else :
		TASK_DATA[task_id]['trans_result'] = []

async def manual_trans_task(task_id, texts) :
	if texts :
		TASK_DATA[task_id]['trans_request'] = [{'s': txt, 't': ''} for txt in texts]
	else :
		TASK_DATA[task_id]['trans_result'] = []
		print('manual translation complete')

@routes.post("/post-translation-result")
async def post_translation_result(request):
	rqjson = (await request.json())
	if 'trans_result' in rqjson and 'task_id' in rqjson :
		task_id = rqjson['task_id']
		if task_id in TASK_DATA :
			trans_result = [r['t'] for r in rqjson['trans_result']]
			TASK_DATA[task_id]['trans_result'] = trans_result
			while True :
				await asyncio.sleep(0.1)
				if TASK_STATES[task_id] in ['error', 'error-lang'] :
					ret = web.json_response({'task_id' : task_id, 'status': 'failed'})
					break
				if TASK_STATES[task_id] == 'finished' :
					ret = web.json_response({'task_id' : task_id, 'status': 'successful'})
					break
			# remove old tasks
			del TASK_STATES[task_id]
			del TASK_DATA[task_id]
			return ret
	return web.json_response({})

@routes.post("/request-translation-internal")
async def request_translation_internal(request):
	global NONCE
	rqjson = (await request.json())
	if rqjson['nonce'] == NONCE :
		task_id = rqjson['task_id']
		if task_id in TASK_DATA :
			if 'manual' in TASK_DATA[task_id] :
				# manual translation
				asyncio.gather(manual_trans_task(task_id, rqjson['texts']))
			else :
				# using machine trnaslation
				asyncio.gather(machine_trans_task(task_id, rqjson['texts'], TASK_DATA[task_id]['translator'], TASK_DATA[task_id]['tgt']))
	return web.json_response({})

@routes.post("/get-translation-result-internal")
async def get_translation_internal(request):
	global NONCE
	rqjson = (await request.json())
	if rqjson['nonce'] == NONCE :
		task_id = rqjson['task_id']
		if task_id in TASK_DATA :
			if 'trans_result' in TASK_DATA[task_id] :
				return web.json_response({'result': TASK_DATA[task_id]['trans_result']})
	return web.json_response({})

@routes.get("/task-state")
async def get_task_state_async(request):
	task_id = request.query.get('taskid')
	if task_id and task_id in TASK_STATES :
		try :
			ret = web.json_response({'state': TASK_STATES[task_id], 'waiting': QUEUE.index(task_id) + 1})
		except :
			ret = web.json_response({'state': TASK_STATES[task_id], 'waiting': 0})
		if TASK_STATES[task_id] in ['finished', 'error', 'error-lang'] :
			# remove old tasks
			del TASK_STATES[task_id]
			del TASK_DATA[task_id]
		return ret
	return web.json_response({'state': 'error'})

@routes.post("/task-update-internal")
async def post_task_update_async(request):
	global NONCE
	rqjson = (await request.json())
	if rqjson['nonce'] == NONCE :
		task_id = rqjson['task_id']
		if task_id in TASK_STATES :
			TASK_STATES[task_id] = rqjson['state']
			print(f'Task state {task_id} to {TASK_STATES[task_id]}')
	return web.json_response({})

@routes.post("/submit")
async def submit_async(request):
	data = await request.post()
	size = ''
	selected_translator = 'youdao'
	target_language = 'CHS'
	if 'tgt_lang' in data :
		target_language = data['tgt_lang'].upper()
		if target_language not in VALID_LANGUAGES :
			target_language = 'CHS'
	if 'translator' in data :
		selected_translator = data['translator'].lower()
		if selected_translator not in ['youdao', 'baidu'] :
			selected_translator = 'youdao'
	if 'size' in data :
		size = data['size'].upper()
		if size not in ['S', 'M', 'L', 'X'] :
			size = ''
	if 'file' in data :
		file_field = data['file']
		content = file_field.file.read()
	elif 'url' in data :
		from aiohttp import ClientSession
		async with ClientSession() as session:
			async with session.get(data['url']) as resp:
				if resp.status == 200 :
					content = await resp.read()
				else :
					return web.json_response({'status' : 'failed'})
	else :
		return web.json_response({'status' : 'failed'})
	try :
		img = convert_img(Image.open(io.BytesIO(content)))
	except :
		return web.json_response({'status' : 'failed'})
	task_id = crypto_utils.rand_bytes(16).hex() + size
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'translator': selected_translator, 'tgt': target_language}
	TASK_STATES[task_id] = 'pending'
	return web.json_response({'task_id' : task_id, 'status': 'successful'})

@routes.post("/manual-translate")
async def manual_translate_async(request):
	data = await request.post()
	size = ''
	if 'size' in data :
		size = data['size'].upper()
		if size not in ['S', 'M', 'L', 'X'] :
			size = ''
	if 'file' in data :
		file_field = data['file']
		content = file_field.file.read()
	elif 'url' in data :
		from aiohttp import ClientSession
		async with ClientSession() as session:
			async with session.get(data['url']) as resp:
				if resp.status == 200 :
					content = await resp.read()
				else :
					return web.json_response({'status' : 'failed'})
	else :
		return web.json_response({'status' : 'failed'})
	try :
		img = convert_img(Image.open(io.BytesIO(content)))
	except :
		return web.json_response({'status' : 'failed'})
	task_id = crypto_utils.rand_bytes(16).hex() + size
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'manual': True}
	TASK_STATES[task_id] = 'pending'
	while True :
		await asyncio.sleep(0.1)
		if 'trans_request' in TASK_DATA[task_id] :
			return web.json_response({'task_id' : task_id, 'status': 'pending', 'trans_result': TASK_DATA[task_id]['trans_request']})
		if TASK_STATES[task_id] in ['error', 'error-lang'] :
			break
		if TASK_STATES[task_id] == 'finished' :
			# no texts detected
			return web.json_response({'task_id' : task_id, 'status': 'successful'})
	return web.json_response({'task_id' : task_id, 'status': 'failed'})

app.add_routes(routes)

async def start_async_app():
	# schedule web server to run
	global NONCE
	NONCE = sys.argv[1]
	port = int(sys.argv[2])
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, '127.0.0.1', port)
	await site.start()
	print(f"Serving up app on 127.0.0.1:{port}")
	return runner, site

loop = asyncio.get_event_loop()
runner, site = loop.run_until_complete(start_async_app())

try:
	loop.run_forever()
except KeyboardInterrupt as err:
	loop.run_until_complete(runner.cleanup())

