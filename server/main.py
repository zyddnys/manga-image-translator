import io
import os
import secrets
import shutil
import signal
import subprocess
import sys
from argparse import Namespace

from fastapi import FastAPI, Request, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pathlib import Path

from pydantic import BaseModel

from server.instance import ExecutorInstance, executor_instances
from server.myqueue import task_queue
from server.request_extraction import get_ctx, while_streaming
from server.to_json import to_json, Translation

app = FastAPI()
nonce = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequestForm(BaseModel):
    """This request can be a multipart or a json request"""
    image: UploadFile
    """can be a url, base64 encoded image or a multipart image"""
    config: str
    """in case it is a multipart this needs to be a string(json.stringify)"""

@app.post("/register", response_description="no response")
async def register_instance(instance: ExecutorInstance, req: Request, req_nonce: str = Header(alias="X-Nonce")):
    if req_nonce != nonce:
        raise HTTPException(401, detail="Invalid nonce")
    instance.ip = req.client.host
    executor_instances.register(instance)

def transform_to_image(ctx):
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()

def transform_to_json(ctx):
    return str(to_json(ctx)).encode("utf-8")


@app.post("/translate/json", response_model=list[Translation], response_description="json strucure inspired by the ichigo translator extension")
async def json(req: Request):
    ctx = await get_ctx(req)
    return JSONResponse(content=to_json(ctx))

@app.post("/translate/bytes", response_class=StreamingResponse, response_description="custom byte structure following the stream encoding, but with json first and then the image bytes as chunks")
async def bytes(req: Request):
    ctx = await get_ctx(req)

@app.post("/translate/image", response_description="the result image", response_class=StreamingResponse)
async def image(req: Request) -> StreamingResponse:
    ctx = await get_ctx(req)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/translate/json/stream", response_class=StreamingResponse, response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_json(req: Request) -> StreamingResponse:
    return await while_streaming(req, transform_to_json)

@app.post("/translate/bytes/stream", response_class=StreamingResponse, response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_bytes(req: Request)-> StreamingResponse:
    return await while_streaming(req, transform_to_image)

@app.post("/translate/image/stream", response_class=StreamingResponse, response_description="A stream over elements with strucure(1byte status, 4 byte size, n byte data) status code are 0,1,2,3,4 0 is result data, 1 is progress report, 2 is error, 3 is waiting queue position, 4 is waiting for translator instance")
async def stream_image(req: Request) -> StreamingResponse:
    return await while_streaming(req, transform_to_image)

@app.post("/queue-size", response_model=int)
async def queue_size() -> int:
    return len(task_queue.queue)

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_file = Path("index.html")
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)

@app.get("/manual", response_class=HTMLResponse)
async def manual():
    html_file = Path("manual.html")
    html_content = html_file.read_text()
    return HTMLResponse(content=html_content)

def generate_nonce():
    return secrets.token_hex(16)

def start_translator_client_proc(host: str, port: int, nonce: str, params: Namespace):
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        'shared',
        '--host', host,
        '--port', str(port),
        '--nonce', nonce,
    ]
    if params.use_gpu:
        cmds.append('--use-gpu')
    if params.use_gpu_limited:
        cmds.append('--use-gpu-limited')
    if params.ignore_errors:
        cmds.append('--ignore-errors')
    if params.verbose:
        cmds.append('--verbose')
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base_path)
    proc = subprocess.Popen(cmds, cwd=parent)
    executor_instances.register(ExecutorInstance(ip=host, port=port))

    def handle_exit_signals(signal, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit_signals)
    signal.signal(signal.SIGTERM, handle_exit_signals)

    return proc

def prepare(args):
    global nonce
    if args.nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())
    else:
        nonce = args.nonce
    if args.start_instance:
        return start_translator_client_proc(args.host, args.port + 1, nonce, args)
    folder_name= "upload-cache"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

#todo: restart if crash
#todo: cache results
#todo: cleanup cache

#todo: add docs
#todo: enable config in html pages

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()
    args.start_instance = True
    proc = prepare(args)
    print("Nonce: "+nonce)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception:
        if proc:
            proc.terminate()
