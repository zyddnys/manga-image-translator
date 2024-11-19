import asyncio
from typing import List, Dict


class TaskQueue:
    def __init__(self):
        self.queue: List[Dict] = []
        self.queue_event: asyncio.Event = asyncio.Event()

    def add_task(self, task):
        self.queue.append(task)

    def get_pos(self, task):
        return self.queue.index(task)

    def update_event(self):
        self.queue_event.set()
        self.queue_event.clear()

    def remove(self, task):
        self.queue.remove(task)
        self.update_event()

    async def wait_for_event(self):
        await self.queue_event.wait()