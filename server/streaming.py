import pickle

async def stream(messages):
    while True:
        message = await messages.get()
        yield message
        if message[0] == 0 or message[0] == 2:
            break

def notify(code, data, transform_to_bytes, messages):
    if code == 0:
        result_bytes = transform_to_bytes(pickle.loads(data))
        encoded_result = b"" + len(result_bytes).to_bytes(4, 'big') + result_bytes
        messages.put_nowait(encoded_result)
    else:
        result_bytes = str(data).encode("utf-8")
        encoded_result =code.to_bytes(1, 'big') + len(result_bytes).to_bytes(4, 'big') + result_bytes
        messages.put_nowait(encoded_result)