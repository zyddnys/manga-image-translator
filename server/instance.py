import pickle
from asyncio import Event, Lock
from typing import List, Optional

from PIL import Image
from pydantic import BaseModel

from manga_translator import Config
from server.sent_data import fetch_data_stream, NotifyType
from fastapi import Response


class ExecutorInstance(BaseModel):
    ip: str
    port: int
    busy: bool = False

    def free_executor(self):
        self.busy = False

    async def sent(self) -> Response:
        pass

    async def sent_stream(self, image: Image, config: Config, sender: NotifyType):
        await fetch_data_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate", image, config, sender)

class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = []
        self.lock: Lock = Lock()
        self.event = Event()

    def register(self, instance: ExecutorInstance):
        self.list.append(instance)

    def free_executors(self) -> int:
        return len([item for item in self.list if not item.busy])

    async def _find_instance(self):
        while True:
            instance = next((x for x in self.list if x.busy == False), None)
            if instance is not None:
                return instance
            #todo: cricial error: warn should never happen
            await self.event.wait()

    async def find_executor(self) -> ExecutorInstance:
        async with self.lock:  # Using async with for lock management
            instance = await self._find_instance()
            instance.busy = True
            return instance

    def free_executor(self, instance: ExecutorInstance):
        instance.free_executor()
        self.event.set()
        self.event.clear()

def example_notify(a: int, b) -> None:
    if a == 0:
        print(pickle.loads(b))
    else:
        print(f"Notify called with a={a} and b={b}")

async def main():
    executor = ExecutorInstance(ip="127.0.0.1", port=5003)

    image = Image.open("../imgs/232264684-5a7bcf8e-707b-4925-86b0-4212382f1680.png")
    config = Config()

    await executor.sent_stream(image, config, example_notify)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())