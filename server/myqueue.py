import asyncio
from typing import List, Dict

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse

from manga_translator import Context
from server.instance import executor_instances
from server.sent_data_internal import NotifyType

class TaskQueue:
    def __init__(self):
        self.queue: List[Context] = []
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

task_queue = TaskQueue()

async def is_client_disconnected(request: Request) -> bool:
    if await request.is_disconnected():
        return True
    return False

async def wait_in_queue(task, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    while True:
        queue_pos = task_queue.get_pos(task)
        if notify:
            notify(3, str(queue_pos).encode('utf-8'))
        if queue_pos < executor_instances.free_executors():
            if await is_client_disconnected(task.req):
                task_queue.remove(task)
                task_queue.update_event()
                if notify:
                    return
                else:
                    raise HTTPException(500, detail="User is no longer connected") #just for the logs
            instance = await executor_instances.find_executor()
            task_queue.remove(task)
            if notify:
                notify(4, b"")
            if notify:
                await instance.sent_stream(task.image, task.config, notify)
            else:
                result = await instance.sent(task.image, task.config)

            executor_instances.free_executor(instance)
            task_queue.update_event()

            if notify:
                return
            else:
                return result
        else:
            if await is_client_disconnected(task.req):
                task_queue.remove(task)
                task_queue.update_event()
                if notify:
                    return
                else:
                    raise HTTPException(500, detail="User is no longer connected") #just for the logs
            await task_queue.wait_for_event()