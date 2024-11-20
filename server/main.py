import io
import os
import secrets
import subprocess
import sys

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

from server.instance import ExecutorInstance, executor_instances
from server.myqueue import task_queue
from server.request_extraction import get_ctx, while_streaming
from server.to_json import to_json

app = FastAPI()
nonce = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register")
async def register_instance(instance: ExecutorInstance, request: Request):
    req_nonce = request.headers.get('X-Nonce')
    if req_nonce != nonce:
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = request.client.host
    executor_instances.register(instance)
    return {"code": 0}

def transform_to_image(ctx):
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

def transform_to_json(ctx):
    return str(to_json(ctx)).encode("utf-8")

@app.post("/translate/json")
async def json(req: Request):
    ctx = await get_ctx(req)
    json = to_json(ctx)
    return JSONResponse(content=json)

@app.post("/translate/bytes")
async def bytes(req: Request):
    ctx = await get_ctx(req)

@app.post("/translate/image")
async def image(req: Request):
    ctx = await get_ctx(req)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/json/stream")
async def stream_json(req: Request):
    return await while_streaming(req, transform_to_json)

@app.post("/translate/bytes/stream")
async def stream_bytes(req: Request):
    return await while_streaming(req, transform_to_image)

@app.post("/translate/image/stream")
async def stream_image(req: Request):
    return await while_streaming(req, transform_to_image)

@app.post("/queue-size")
async def queue_size() -> int:
    return len(task_queue.queue)

@app.post("/")
async def index():
    # todo:ui.html
    pass

@app.post("/manual")
async def manual():
    # todo:manual.html
    pass

def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: dict):
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        '--mode', 'shared',
        '--host', host,
        '--port', str(port),
        '--nonce', nonce,
        '--no-report'
    ]
    if params.get('use_gpu', False):
        cmds.append('--use-gpu')
    if params.get('use_gpu_limited', False):
        cmds.append('--use-gpu-limited')
    if params.get('ignore_errors', False):
        cmds.append('--ignore-errors')
    if params.get('verbose', False):
        cmds.append('--verbose')
    #todo: cwd
    proc = subprocess.Popen(cmds, cwd=BASE_PATH)
    executor_instances.register(ExecutorInstance(ip=host, port=port))
    return proc

def prepare(args):
    global nonce
    if args.get("nonce", None) is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    else:
        nonce = args.get("nonce", None)
    if args.get("start_instance", None):
        start_translator_client_proc(args.get("host", "0.0.0.0"), args.get("port",8000) + 1, nonce, args)

#todo: restart if crash
#todo: cache results
#todo: cleanup cache
#todo: store images while in queue
#todo: add docs

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()
    prepare(args)
    print("Nonce: "+nonce)
    uvicorn.run(app, host=args.host, port=args.port)
