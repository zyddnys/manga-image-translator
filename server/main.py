import asyncio
import builtins
import io
import re
from base64 import b64decode
from typing import Union

import requests
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

from manga_translator import Config, Context
from server.instance import ExecutorInstance, Executors
from server.myqueue import TaskQueue
from server.sent_data import NotifyType
from server.to_json import to_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
executor_instances: Executors = Executors()
task_queue = TaskQueue()


@app.post("/register")
async def register_instance(instance: ExecutorInstance, request: Request):
    instance.ip = request.client.host
    executor_instances.register(instance)
    return {"code": 0}


async def to_pil_image(image: Union[str, bytes]) -> Image.Image:
    try:
        if isinstance(image, builtins.bytes):
            image = Image.open(io.BytesIO(image))
            return image
        else:
            if re.match(r'^data:image/.+;base64,', image):
                value = image.split(',', 1)[1]
                image_data = b64decode(value)
                image = Image.open(io.BytesIO(image_data))
                return image
            else:
                response = requests.get(image)
                image = Image.open(io.BytesIO(response.content))
                return image
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


async def multi_content_type(request: Request):
    content_type = request.headers.get("content-type")

    if content_type and content_type.startswith("multipart/form-data"):
        form = await request.form()
        config = form.get("config", "{}")
        image = form.get("image")
        image_content = await image.read()
        config = Config.parse_raw(config)
        return config, image_content
    elif content_type and content_type.startswith("application/json"):
        body = await request.json()
        config = Config(**body.get("config", {}))
        image = body.get("image")
        return config, image

    else:
        raise HTTPException(status_code=400, detail="Unsupported Content-Type")


async def wait(task, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    while True:
        queue_pos = task_queue.get_pos(task)
        if notify:
            notify(3, queue_pos)
        if queue_pos < executor_instances.free_executors():
            instance = await executor_instances.find_executor()
            task_queue.remove(task)
            if notify:
                notify(4, 0)
            if notify:
                await instance.sent_stream(task.image, task.config, notify)
            else:
                result = await instance.sent(task.image, task.config)

            executor_instances.free_executor(instance)

            if notify:
                return
            else:
                return result
        else:
            if queue_pos == 0:
                raise HTTPException(500, detail="No translator registered")
            await task_queue.wait_for_event()


async def stream(messages):
    while True:
        message = await messages.get()
        yield message
        if message[0] == 0 or message[0] == 2:
            break

def notify(code, data, transform_to_bytes, messages):
    if code == 0:
        result_bytes = transform_to_bytes(data)
        encoded_result = b"" + len(result_bytes).to_bytes(4, 'big') + result_bytes
        messages.put_nowait(encoded_result)
    else:
        result_bytes = str(data).encode("utf-8")
        encoded_result =code.to_bytes(1, 'big') + len(result_bytes).to_bytes(4, 'big') + result_bytes
        messages.put_nowait(encoded_result)

def transform_to_image(data):
    return b""

@app.post("/json")
async def json(req: Request):
    data, img = await multi_content_type(req)
    ctx = Context()

    ctx.image = await to_pil_image(img)
    ctx.config = data
    task_queue.add_task(ctx)

    data = await wait(ctx, None)
    json = to_json(data)
    return JSONResponse(content=json)

@app.post("/bytes")
async def bytes(req: Request):
    data, img = await multi_content_type(req)
    ctx = Context()

    ctx.image = await to_pil_image(img)
    ctx.config = data
    task_queue.add_task(ctx)
    data = await wait((data, img), None)


@app.post("/image")
async def image(req: Request):
    data, img = await multi_content_type(req)
    ctx = Context()

    ctx.image = await to_pil_image(img)
    ctx.config = data
    task_queue.add_task(ctx)

    data = await wait((data, img), None)
    img_byte_arr = io.BytesIO()
    data.result.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/stream_json")
async def stream_json(req: Request):
    data, img = await multi_content_type(req)
    img = await to_pil_image(img)

    messages = asyncio.Queue()

    def example_notify(code: int, data) -> None:
        notify(code, data, transform_to_image, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait((data, img), example_notify))
    return streaming_response

@app.post("/stream_bytes")
async def stream_bytes(req: Request):
    data, img = await multi_content_type(req)
    img = await to_pil_image(img)

    messages =  asyncio.Queue()

    def example_notify(code: int, data) -> None:
        notify(code, data, transform_to_image, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait((data, img), example_notify))
    return streaming_response

@app.post("/stream_image")
async def stream_image(req: Request):
    data, img = await multi_content_type(req)
    img = await to_pil_image(img)

    messages =  asyncio.Queue()

    def example_notify(code: int, data) -> None:
        notify(code, data, transform_to_image, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait((data, img), example_notify))
    return streaming_response

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()
    uvicorn.run(app, host=args.host, port=args.port)
