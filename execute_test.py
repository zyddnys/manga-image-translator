import asyncio
import json
import pickle
import requests
from PIL import Image

async def execute_method(method_name, attributes):
    url = f"http://127.0.0.1:5003/execute/{method_name}"
    headers = {'Content-Type': 'application/octet-stream'}

    response = requests.post(url, data=pickle.dumps(attributes), headers=headers, stream=True)

    if response.status_code == 200:
        buffer = b''
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                buffer += chunk
                while True:
                    if len(buffer) >= 5:
                        status = int.from_bytes(buffer[0:1], byteorder='big')
                        expected_size = int.from_bytes(buffer[1:5], byteorder='big')
                        if len(buffer) >= 5 + expected_size:
                            data = buffer[5:5 + expected_size]
                            if status == 0:
                                print("data", pickle.loads(data))
                            elif status == 1:
                                print("log", data)
                            elif status == 2:
                                print("error", data)
                            buffer = buffer[5 + expected_size:]
                        else:
                            break
                    else:
                        break
    else:
        print(json.loads(response.content))



if __name__ == '__main__':
    image = Image.open("../imgs/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png")
    attributes = {"image": image, "params": {"translator": "none", "inpainter": "none"}}
    asyncio.run(execute_method("translate", attributes))