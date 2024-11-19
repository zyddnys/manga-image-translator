import asyncio
import io

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

from server.instance import ExecutorInstance, executor_instances
from server.myqueue import wait_in_queue
from server.request_extraction import multi_content_type, to_pil_image, get_ctx
from server.streaming import notify, stream
from server.to_json import to_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register")
async def register_instance(instance: ExecutorInstance, request: Request):
    instance.ip = request.client.host
    executor_instances.register(instance)
    return {"code": 0}

def transform_to_image(data):
    return b""

@app.post("/json")
async def json(req: Request):
    ctx = await get_ctx(req)
    json = to_json(ctx)
    return JSONResponse(content=json)

@app.post("/bytes")
async def bytes(req: Request):
    ctx = await get_ctx(req)


@app.post("/image")
async def image(req: Request):
    ctx = await get_ctx(req)
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
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
    asyncio.create_task(wait_in_queue((data, img), example_notify))
    return streaming_response

@app.post("/stream_bytes")
async def stream_bytes(req: Request):
    data, img = await multi_content_type(req)
    img = await to_pil_image(img)

    messages =  asyncio.Queue()

    def example_notify(code: int, data) -> None:
        notify(code, data, transform_to_image, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait_in_queue((data, img), example_notify))
    return streaming_response

@app.post("/stream_image")
async def stream_image(req: Request):
    data, img = await multi_content_type(req)
    img = await to_pil_image(img)

    messages =  asyncio.Queue()

    def example_notify(code: int, data) -> None:
        notify(code, data, transform_to_image, messages)

    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait_in_queue((data, img), example_notify))
    return streaming_response

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments

    args = parse_arguments()
    uvicorn.run(app, host=args.host, port=args.port)
