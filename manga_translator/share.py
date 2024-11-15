import asyncio
import pickle
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Path, Request, Response
from pydantic import BaseModel
import inspect

from manga_translator import MangaTranslator

class MethodCall(BaseModel):
    method_name: str
    attributes: bytes

class MangaShare(MangaTranslator):
    def __init__(self, params: dict = None):
        import nest_asyncio
        nest_asyncio.apply()
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = int(params.get('port', '5003'))
        self.nonce = params.get('nonce', None)
        self.lock = Lock()

    async def listen(self, translation_params: dict = None):
        app = FastAPI()

        @app.get("/is_locked")
        async def is_locked():
            if self.lock.locked():
                return {"locked": True}
            return {"locked": False}

        @app.post("/execute/{method_name}")
        async def execute_method(request: Request, method_name: str = Path(...)):
            if self.nonce:
                nonce = request.headers.get('X-Nonce')
                if nonce != self.nonce:
                    raise HTTPException(401, detail="Nonce does not match")

            if not self.lock.acquire(blocking=False):
                raise HTTPException(status_code=429, detail="some Method is already being executed.")
            if method_name == "listen" or method_name.startswith("__"):
                raise HTTPException(status_code=403, detail="These functions are not allowed to be executed remotely")
            method = getattr(self, method_name, None)
            if not method:
                raise HTTPException(status_code=404, detail="Method not found")
            attributes_bytes = await request.body()
            attributes = pickle.loads(attributes_bytes)
            sig = inspect.signature(method)
            expected_args = set(sig.parameters.keys())
            provided_args = set(attributes.keys())

            if expected_args != provided_args:
                raise HTTPException(status_code=400, detail="Incorrect number or names of arguments")

            try:
                if asyncio.iscoroutinefunction(method):
                    result = await method(**attributes)
                else:
                    result = method(**attributes)
                result_bytes = pickle.dumps(result)
                return Response(content=result_bytes, media_type="application/octet-stream")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        uvicorn.run(app, host=self.host, port=self.port)
