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
from io import BytesIO

from collections import deque

from translators import VALID_LANGUAGES, dispatch as run_translation

VALID_DETECTORS = set(['default', 'ctd'])
VALID_DIRECTIONS = set(['auto', 'horizontal'])

NONCE = ''
QUEUE = deque()
TASK_DATA = {}
TASK_STATES = {}

app = web.Application(client_max_size = 1024 * 1024 * 10)
routes = web.RouteTableDef()

@routes.get("/")
async def index_async(request) :
	with open('ui.html', 'r', encoding='utf8') as fp :
		return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/result/{taskid}")
async def result_async(request) :
        im = Image.open("result/" + request.match_info.get('taskid') + "/final.png")
        stream = BytesIO()
        im.save(stream, "PNG")
        return web.Response(body=stream.getvalue(), content_type='image/png')

async def handle_post(request) :
	data = await request.post()
	size = ''
	selected_translator = 'youdao'
	target_language = 'CHS'
	detector = 'default'
	direction = 'auto'
	if 'tgt_lang' in data :
		target_language = data['tgt_lang'].upper()
		if target_language not in VALID_LANGUAGES :
			target_language = 'CHS'
	if 'detector' in data :
		detector = data['detector'].lower()
		if detector not in VALID_DETECTORS :
			detector = 'default'
	if 'dir' in data :
		direction = data['dir'].lower()
		if direction not in VALID_DIRECTIONS :
			direction = 'auto'
	if 'translator' in data :
		selected_translator = data['translator'].lower()
		if selected_translator not in ['youdao', 'baidu', 'google', 'deepl', 'null'] :
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
		img = Image.open(io.BytesIO(content))
	except :
		return web.json_response({'status' : 'failed'})
	return img, size, selected_translator, target_language, detector, direction

@routes.post("/run")
async def run_async(request) :
	x = await handle_post(request)
	if isinstance(x, tuple) :
		img, size, selected_translator, target_language, detector, direction = x
	else :
		return x
	task_id = crypto_utils.rand_bytes(16).hex()
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'size': size, 'translator': selected_translator, 'tgt': target_language, 'detector': detector, 'direction': direction}
	TASK_STATES[task_id] = 'pending'
	while True :
		await asyncio.sleep(0.05)
		state = TASK_STATES[task_id]
		if state == 'finished' :
			break
	return web.json_response({'task_id' : task_id, 'status': 'successful'})


@routes.get("/task-internal")
async def get_task_async(request) :
	global NONCE
	if request.rel_url.query['nonce'] == NONCE :
		if len(QUEUE) > 0 :
			task_id = QUEUE.popleft()
			data = TASK_DATA[task_id]
			return web.json_response({'task_id': task_id, 'data': data})
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
async def post_translation_result(request) :
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
async def request_translation_internal(request) :
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
async def get_translation_internal(request) :
	global NONCE
	rqjson = (await request.json())
	if rqjson['nonce'] == NONCE :
		task_id = rqjson['task_id']
		if task_id in TASK_DATA :
			if 'trans_result' in TASK_DATA[task_id] :
				return web.json_response({'result': TASK_DATA[task_id]['trans_result']})
	return web.json_response({})

@routes.get("/task-state")
async def get_task_state_async(request) :
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
async def post_task_update_async(request) :
	global NONCE
	rqjson = (await request.json())
	if rqjson['nonce'] == NONCE :
		task_id = rqjson['task_id']
		if task_id in TASK_STATES :
			TASK_STATES[task_id] = rqjson['state']
			print(f'Task state {task_id} to {TASK_STATES[task_id]}')
	return web.json_response({})

@routes.post("/submit")
async def submit_async(request) :
	x = await handle_post(request)
	if isinstance(x, tuple) :
		img, size, selected_translator, target_language, detector, direction = x
	else :
		return x
	task_id = crypto_utils.rand_bytes(16).hex()
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'size': size, 'translator': selected_translator, 'tgt': target_language, 'detector': detector, 'direction': direction}
	TASK_STATES[task_id] = 'pending'
	return web.json_response({'task_id' : task_id, 'status': 'successful'})

@routes.post("/manual-translate")
async def manual_translate_async(request) :
	x = await handle_post(request)
	if isinstance(x, tuple) :
		img, size, selected_translator, target_language, detector, direction = x
	else :
		return x
	task_id = crypto_utils.rand_bytes(16).hex()
	os.makedirs(f'result/{task_id}/', exist_ok=True)
	img.save(f'result/{task_id}/input.png')
	QUEUE.append(task_id)
	TASK_DATA[task_id] = {'size': size, 'manual': True, 'detector': detector, 'direction': direction}
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

async def start_async_app() :
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

if __name__ == '__main__' :
	loop = asyncio.get_event_loop()
	runner, site = loop.run_until_complete(start_async_app())

	try:
		loop.run_forever()
	except KeyboardInterrupt as err :
		loop.run_until_complete(runner.cleanup())

