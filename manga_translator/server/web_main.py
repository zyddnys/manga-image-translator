import io
import os
import sys
import time
import asyncio
import threading
from PIL import Image
from oscrypto import util as crypto_utils
from aiohttp import web
from io import BytesIO
from imagehash import phash
from collections import deque

SERVER_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.dirname(os.path.dirname(SERVER_DIR_PATH))

VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'PLK': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
}
VALID_DETECTORS = set(['default', 'ctd'])
VALID_DIRECTIONS = set(['auto', 'h', 'v'])
VALID_TRANSLATORS = set(['youdao', 'baidu', 'google', 'deepl', 'papago', 'offline', 'none', 'original'])

MAX_ONGOING_TASKS = 1
MAX_IMAGE_SIZE_PX = 8000**2

# Time to wait for translator client before automatically shutting down
TRANSLATOR_CLIENT_TIMEOUT = 20

# Time to wait for webclient to send a request to /task-state request
# before that web clients task gets removed from the queue
WEB_CLIENT_TIMEOUT = 10

NUM_ONGOING_TASKS = 0
NONCE = ''
QUEUE = deque()
TASK_DATA = {}
TASK_STATES = {}

app = web.Application(client_max_size = 1024 * 1024 * 50)
routes = web.RouteTableDef()


def constant_compare(a, b):
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        return False
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0

@routes.get("/")
async def index_async(request):
    with open(os.path.join(SERVER_DIR_PATH, 'ui.html'), 'r', encoding='utf8') as fp:
        return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/manual")
async def index_async(request):
    with open(os.path.join(SERVER_DIR_PATH, 'manual.html'), 'r', encoding='utf8') as fp:
        return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/result/{taskid}")
async def result_async(request):
    im = Image.open("result/" + request.match_info.get('taskid') + "/final.png")
    stream = BytesIO()
    im.save(stream, "PNG")
    return web.Response(body=stream.getvalue(), content_type='image/png')

@routes.get("/queue-size")
async def queue_size_async(request):
    return web.json_response({'size' : len(QUEUE)})

async def handle_post(request):
    data = await request.post()
    detection_size = None
    selected_translator = 'youdao'
    target_language = 'CHS'
    detector = 'default'
    direction = 'auto'
    if 'tgt_lang' in data:
        target_language = data['tgt_lang'].upper()
        # TODO: move dicts to their own files to reduce load time
        if target_language not in VALID_LANGUAGES:
            target_language = 'CHS'
    if 'detector' in data:
        detector = data['detector'].lower()
        if detector not in VALID_DETECTORS:
            detector = 'default'
    if 'direction' in data:
        direction = data['direction'].lower()
        if direction not in VALID_DIRECTIONS:
            direction = 'auto'
    if 'translator' in data:
        selected_translator = data['translator'].lower()
        if selected_translator not in VALID_TRANSLATORS:
            selected_translator = 'youdao'
    if 'size' in data:
        size_text = data['size'].upper()
        if size_text == 'S':
            detection_size = 1024
        elif size_text == 'M':
            detection_size = 1536
        elif size_text == 'L':
            detection_size = 2048
        elif size_text == 'X':
            detection_size = 2560
    if 'file' in data:
        file_field = data['file']
        content = file_field.file.read()
    elif 'url' in data:
        from aiohttp import ClientSession
        async with ClientSession() as session:
            async with session.get(data['url']) as resp:
                if resp.status == 200:
                    content = await resp.read()
                else:
                    return web.json_response({'status': 'error'})
    else:
        return web.json_response({'status': 'error'})
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width * img.height > MAX_IMAGE_SIZE_PX:
            return web.json_response({'status': 'error-too-large'})
    except Exception:
        return web.json_response({'status': 'error-img-corrupt'})
    return img, detection_size, selected_translator, target_language, detector, direction

@routes.post("/run")
async def run_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = f'{phash(img, hash_size = 16)}-{size}-{selected_translator}-{target_language}-{detector}-{direction}'
    print(f'New `run` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.png'):
        return web.json_response({'task_id' : task_id, 'status': 'successful'})
    # elif os.path.exists(f'result/{task_id}'):
    #     # either image is being processed or error occurred 
    #     if task_id not in TASK_STATES:
    #         # error occurred
    #         return web.json_response({'state': 'error'})
    else:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': time.time(),
        }
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
    while True:
        await asyncio.sleep(0.1)
        if task_id not in TASK_STATES:
            break
        state = TASK_STATES[task_id]
        if state['finished']:
            break
    return web.json_response({'task_id': task_id, 'status': 'successful' if state['finished'] else state['info']})


@routes.get("/task-internal")
async def get_task_async(request):
    """
    Called by the translator to get a translation task.
    """
    global NONCE, NUM_ONGOING_TASKS
    if constant_compare(request.rel_url.query.get('nonce'), NONCE):
        if len(QUEUE) > 0 and NUM_ONGOING_TASKS < MAX_ONGOING_TASKS:
            task_id = QUEUE.popleft()
            if task_id in TASK_DATA:
                data = TASK_DATA[task_id]
                if not TASK_DATA[task_id].get('manual', False):
                    NUM_ONGOING_TASKS += 1
                return web.json_response({'task_id': task_id, 'data': data})
            else:
                return web.json_response({})
        else:
            return web.json_response({})
    return web.json_response({})

# async def machine_trans_task(task_id, texts, translator = 'youdao', target_language = 'CHS'):
#     print('translator', translator)
#     print('target_language', target_language)
#     if task_id not in TASK_DATA:
#         TASK_DATA[task_id] = {}
#     if texts:
#         success = False
#         for _ in range(10):
#             try:
#                 TASK_DATA[task_id]['trans_result'] = await asyncio.wait_for(dispatch_translation(translator, 'auto', target_language, texts), timeout = 15)
#                 success = True
#                 break
#             except Exception as ex:
#                 continue
#         if not success:
#             TASK_DATA[task_id]['trans_result'] = 'error'
#     else:
#         TASK_DATA[task_id]['trans_result'] = []

async def manual_trans_task(task_id, texts):
    if task_id not in TASK_DATA:
        TASK_DATA[task_id] = {}
    if texts:
        TASK_DATA[task_id]['trans_request'] = [{'s': txt, 't': ''} for txt in texts]
    else:
        TASK_DATA[task_id]['trans_result'] = []
        print('manual translation complete')

@routes.post("/post-translation-result")
async def post_translation_result(request):
    rqjson = (await request.json())
    if 'trans_result' in rqjson and 'task_id' in rqjson:
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            trans_result = [r['t'] for r in rqjson['trans_result']]
            TASK_DATA[task_id]['trans_result'] = trans_result
            while True:
                await asyncio.sleep(0.1)
                if TASK_STATES[task_id]['info'].startswith('error'):
                    ret = web.json_response({'task_id': task_id, 'status': 'error'})
                    break
                if TASK_STATES[task_id]['finished']:
                    ret = web.json_response({'task_id': task_id, 'status': 'successful'})
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
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if TASK_DATA[task_id].get('manual', False):
                # manual translation
                asyncio.gather(manual_trans_task(task_id, rqjson['texts']))
            # else:
            #     # using machine translation
            #     asyncio.gather(machine_trans_task(task_id, rqjson['texts'], TASK_DATA[task_id]['translator'], TASK_DATA[task_id]['tgt']))
    return web.json_response({})

@routes.post("/get-translation-result-internal")
async def get_translation_internal(request):
    global NONCE
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if 'trans_result' in TASK_DATA[task_id]:
                return web.json_response({'result': TASK_DATA[task_id]['trans_result']})
    return web.json_response({})

@routes.get("/task-state")
async def get_task_state_async(request):
    """
    Web API for getting the state of an on-going translation task from the website.

    Is periodically called from ui.html. Once it returns a finished state,
    the web client will try to fetch the corresponding image through /result/<task_id>
    """
    task_id = request.query.get('taskid')
    if task_id and task_id in TASK_STATES and task_id in TASK_DATA:
        state = TASK_STATES[task_id]
        res_dict = {
            'state': state['info'],
            'finished': state['finished'],
        }
        try:
            res_dict['waiting'] = QUEUE.index(task_id) + 1
        except Exception:
            res_dict['waiting'] = 0
        res = web.json_response(res_dict)

        # remove old tasks that are 30 minutes old
        now = time.time()
        to_del_task_ids = set()
        for tid, s in TASK_STATES.items():
            if s['finished'] and now - TASK_DATA[tid]['created_at'] > 1800:
                to_del_task_ids.add(tid)
        for tid in to_del_task_ids:
            del TASK_STATES[tid]
            del TASK_DATA[tid]

        return res
    return web.json_response({'state': 'error'})

@routes.post("/task-update-internal")
async def post_task_update_async(request):
    """
    Lets the translator update the task state it is working on.
    """
    global NONCE, NUM_ONGOING_TASKS
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_STATES and task_id in TASK_DATA:
            TASK_STATES[task_id] = {
                'info': rqjson['state'],
                'finished': rqjson['finished'],
            }
            if rqjson['finished'] and not TASK_DATA[task_id].get('manual', False):
                NUM_ONGOING_TASKS -= 1
            print(f'Task state {task_id} to {TASK_STATES[task_id]}')
    return web.json_response({})

@routes.post("/submit")
async def submit_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = f'{phash(img, hash_size = 16)}-{size}-{selected_translator}-{target_language}-{detector}-{direction}'
    print(f'New `submit` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.png'):
        TASK_STATES[task_id] = {
            'info': 'saved',
            'finished': True,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': time.time(),
        }
    # elif os.path.exists(f'result/{task_id}'):
    #     # either image is being processed or error occurred
    #     if task_id not in TASK_STATES:
    #         # error occurred
    #         return web.json_response({'state': 'error'})
    else:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': time.time(),
        }
    return web.json_response({'task_id': task_id, 'status': 'successful'})

@routes.post("/manual-translate")
async def manual_translate_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = crypto_utils.rand_bytes(16).hex()
    print(f'New `manual-translate` task {task_id}')
    os.makedirs(f'result/{task_id}/', exist_ok=True)
    img.save(f'result/{task_id}/input.png')
    QUEUE.append(task_id)
    TASK_DATA[task_id] = {
        'detection_size': size,
        'manual': True,
        'detector': detector,
        'direction': direction,
        'created_at': time.time()
    }
    TASK_STATES[task_id] = {
        'info': 'pending',
        'finished': False,
    }
    while True:
        await asyncio.sleep(1)
        if 'trans_request' in TASK_DATA[task_id]:
            return web.json_response({'task_id' : task_id, 'status': 'pending', 'trans_result': TASK_DATA[task_id]['trans_request']})
        if TASK_STATES[task_id]['info'].startswith('error'):
            break
        if TASK_STATES[task_id]['finished']:
            # no texts detected
            return web.json_response({'task_id' : task_id, 'status': 'successful'})
    return web.json_response({'task_id' : task_id, 'status': 'error'})

app.add_routes(routes)


def generate_nonce():
    return crypto_utils.rand_bytes(16).hex()

class TranslatorClientThread(threading.Thread):

    def __init__(self, host: str, port: int, nonce: str, translation_params: dict = None):
        threading.Thread.__init__(self)
        self.params = translation_params or {}
        self.params['host'] = host
        self.params['port'] = port
        self.params['nonce'] = nonce
        self.daemon = True

    def run(self):
        from ..manga_translator import MangaTranslatorWeb

        translator = MangaTranslatorWeb(self.params)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(translator.listen(self.params))

async def start_async_app(host: str, port: int, nonce: str = None, translation_params: dict = None):
    global NONCE
    # Secret to secure communication between webserver and translator clients
    if nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    NONCE = nonce
    
    # Schedule web server to run
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f'Serving up app on http://{host}:{port}')

    # Create thread with client that will execute translation tasks
    client = TranslatorClientThread(host, port, nonce, translation_params)
    client.start()

    return runner, site

if __name__ == '__main__':
    from ..args import parser

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner, site = loop.run_until_complete(start_async_app(args.host, args.port, vars(args)))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(runner.cleanup())
