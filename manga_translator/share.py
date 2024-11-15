import asyncio
import pickle
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Path, Request, Response
from pydantic import BaseModel
import inspect

from starlette.responses import StreamingResponse

from manga_translator import MangaTranslator

class MethodCall(BaseModel):
    method_name: str
    attributes: bytes

class MangaShare:
    def __init__(self, params: dict = None):
        self.manga = MangaTranslator(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = int(params.get('port', '5003'))
        self.nonce = params.get('nonce', None)

        # each chunk has a structure like this status_code(int/1byte),len(int/4bytes),bytechunk
        # status codes are 0 for result, 1 for progress report, 2 for error
        self.progress_queue = asyncio.Queue()
        self.lock = Lock()

        async def hook(state: str, finished: bool):
            state_data = state.encode("utf-8")
            progress_data = b'\x01' + len(state_data).to_bytes(4, 'big') + state_data
            await self.progress_queue.put(progress_data)
            await asyncio.sleep(0)

        self.manga.add_progress_hook(hook)

    async def progress_stream(self):
        """
        loops until the status is != 1 which is eiter an error or the result
        """
        while True:
            progress = await self.progress_queue.get()
            yield progress
            if progress[0] != 1:
                break

    async def run_method(self, method, **attributes):
        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(**attributes)
            else:
                result = method(**attributes)
            result_bytes = pickle.dumps(result)
            encoded_result = b'\x00' + len(result_bytes).to_bytes(4, 'big') + result_bytes
            await self.progress_queue.put(encoded_result)
        except Exception as e:
            err_bytes = str(e).encode("utf-8")
            encoded_result = b'\x02' + len(err_bytes).to_bytes(4, 'big') + err_bytes
            await self.progress_queue.put(encoded_result)
        finally:
            self.lock.release()


    async def listen(self, translation_params: dict = None):
        app = FastAPI()

        @app.get("/is_locked")
        async def is_locked():
            if self.lock.locked():
                return {"locked": True}
            return {"locked": False}

        @app.post("/execute/{method_name}")
        async def execute_method(request: Request, method_name: str = Path(...)):
            # internal verification
            if self.nonce:
                nonce = request.headers.get('X-Nonce')
                if nonce != self.nonce:
                    raise HTTPException(401, detail="Nonce does not match")
            # only one function at a time
            if not self.lock.acquire(blocking=False):
                raise HTTPException(status_code=429, detail="some Method is already being executed.")
            # block api functions
            if method_name == "listen" or method_name == "run_method" or method_name.startswith("__"):
                raise HTTPException(status_code=403, detail="These functions are not allowed to be executed remotely")

            # find method
            method = getattr(self.manga, method_name, None)
            if not method:
                raise HTTPException(status_code=404, detail="Method not found")

            # load data
            attributes_bytes = await request.body()
            attributes = pickle.loads(attributes_bytes)
            sig = inspect.signature(method)
            expected_args = set(sig.parameters.keys())
            provided_args = set(attributes.keys())

            if expected_args != provided_args:
                raise HTTPException(status_code=400, detail="Incorrect number or names of arguments")

            # streaming response
            streaming_response = StreamingResponse(self.progress_stream(), media_type="application/octet-stream")
            asyncio.create_task(self.run_method(method, **attributes))
            return streaming_response

        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
