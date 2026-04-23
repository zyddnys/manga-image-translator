from typing import List
import pickle
import os
from typing import Mapping, Optional, Callable

import aiohttp
from PIL.Image import Image
from fastapi import HTTPException

from manga_translator import Config

NotifyType = Optional[Callable[[int, Optional[bytes]], None]]


def _with_default_nonce(headers: Mapping[str, str]) -> Mapping[str, str]:
    """Attach X-Nonce for internal calls when available."""
    merged_headers = dict(headers or {})
    nonce = os.getenv('MT_WEB_NONCE')
    if nonce and 'X-Nonce' not in merged_headers:
        merged_headers['X-Nonce'] = nonce
    return merged_headers

async def fetch_data_stream(url, image: Image, config: Config, sender: NotifyType, headers: Mapping[str, str] = {}):
    attributes = {"image": image, "config": config}
    data = pickle.dumps(attributes)
    request_headers = _with_default_nonce(headers)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=request_headers) as response:
            if response.status == 200:
                await process_stream(response, sender)
            else:
                raise HTTPException(response.status, detail=await response.text())

async def fetch_data(url, image: Image, config: Config, headers: Mapping[str, str] = {}):
    attributes = {"image": image, "config": config}
    data = pickle.dumps(attributes)
    request_headers = _with_default_nonce(headers)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=request_headers) as response:
            if response.status == 200:
                return pickle.loads(await response.read())
            else:
                raise HTTPException(response.status, detail=await response.text())

# 原来executorInstance.sent_batch 调fetch接口存在参数不兼容问题，单独实现一个新方法
async def fetch_batch_data(url, images: List[Image], config: Config, batch_size: int, headers: Mapping[str, str] = {}):
    images_with_configs = [(img, config) for img in images]
    data = pickle.dumps({"images_with_configs": images_with_configs, "batch_size": batch_size})
    request_headers = _with_default_nonce(headers)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data, headers=request_headers) as response:
            if response.status == 200:
                return pickle.loads(await response.read())
            else:
                raise HTTPException(response.status, detail=await response.text())

async def process_stream(response, sender: NotifyType):
    buffer = b''

    async for chunk in response.content.iter_any():
        if chunk:
            buffer += chunk
            buffer = handle_buffer(buffer, sender)



def handle_buffer(buffer, sender: NotifyType):
    while len(buffer) >= 5:
        status, expected_size = extract_header(buffer)

        if len(buffer) >= 5 + expected_size:
            data = buffer[5:5 + expected_size]
            sender(status, data)
            buffer = buffer[5 + expected_size:]
        else:
            break
    return buffer


def extract_header(buffer):
    """Extract the status and expected size from the buffer."""
    status = int.from_bytes(buffer[0:1], byteorder='big')
    expected_size = int.from_bytes(buffer[1:5], byteorder='big')
    return status, expected_size

