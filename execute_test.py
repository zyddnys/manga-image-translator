import asyncio
import json
import pickle
import requests
from PIL import Image

async def execute_method(method_name, attributes):
    url = f"http://127.0.0.1:5003/execute/{method_name}"
    headers = {'Content-Type': 'application/octet-stream'}

    response = requests.post(url, data=pickle.dumps(attributes), headers=headers)

    result_bytes = response.content
    if response.status_code == 200:
        result = pickle.loads(result_bytes)
        print(result)
    else:
        print(json.loads(response.content))



if __name__ == '__main__':
    image = Image.open("../imgs/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png")
    attributes = {"image": image, "params": {"translator": "none", "inpainter": "none"}}
    asyncio.run(execute_method("translate", attributes))