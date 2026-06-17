import asyncio
import pickle
import logging

logger = logging.getLogger('manga-translator')

def _emit_progress(state: str, messages: asyncio.Queue):
    logger.info(state)
    state_data = state.encode("utf-8")
    encoded_result = b'\x01' + len(state_data).to_bytes(4, 'big') + state_data
    messages.put_nowait(encoded_result)

async def stream(messages):
    while True:
        message = await messages.get()
        yield message
        if message[0] == 0 or message[0] == 2:
            break

def notify(code: int, data: bytes, transform_to_bytes, messages: asyncio.Queue):
    if code == 0:
        _emit_progress('decoding_result', messages)
        ctx = pickle.loads(data)
        _emit_progress('encoding_image', messages)
        result_bytes = transform_to_bytes(ctx)
        _emit_progress('encoding_image_done', messages)
        encoded_result = b'\x00' + len(result_bytes).to_bytes(4, 'big') + result_bytes
        messages.put_nowait(encoded_result)
    else:
        encoded_result =code.to_bytes(1, 'big') + len(data).to_bytes(4, 'big') + data
        messages.put_nowait(encoded_result)