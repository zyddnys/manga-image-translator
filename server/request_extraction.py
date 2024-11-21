import asyncio
import builtins
import io
import re
from base64 import b64decode
from typing import Union

import requests
from PIL import Image
from fastapi import Request, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from manga_translator import Config, Context
from server.myqueue import task_queue, wait_in_queue, QueueElement
from server.streaming import notify, stream

class TranslateRequest(BaseModel):
    """This request can be a multipart or a json request"""
    image: bytes|str
    """can be a url, base64 encoded image or a multipart image"""
    config: Config
    """in case it is a multipart this needs to be a string(json.stringify)"""

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

async def get_ctx(req: Request):
    data, img = await multi_content_type(req)

    image = await to_pil_image(img)
    task = QueueElement(req, image, data, 0)

    task_queue.add_task(task)

    return await wait_in_queue(task, None)

async def while_streaming(req: Request, transform):
    data, img = await multi_content_type(req)

    image = await to_pil_image(img)
    task = QueueElement(req, image, data, 0)
    task_queue.add_task(task)

    messages = asyncio.Queue()

    def notify_internal(code: int, data: bytes) -> None:
        notify(code, data, transform, messages)
    streaming_response = StreamingResponse(stream(messages), media_type="application/octet-stream")
    asyncio.create_task(wait_in_queue(task, notify_internal))
    return streaming_response