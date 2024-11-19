import asyncio
from typing import List, Dict, Optional, Callable

from fastapi import FastAPI, Request, HTTPException

from server.instance import ExecutorInstance, Executors
from server.myqueue import TaskQueue
from server.sent_data import NotifyType

app = FastAPI()
executor_instances: Executors = Executors()
task_queue = TaskQueue()

@app.post("/register")
async def register_instance(instance: ExecutorInstance, request: Request):
    instance.ip = request.client.host
    executor_instances.register(instance)
    return {"code": 0}



async def wait(task, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    while True:
        queue_pos = task_queue.get_pos(task)
        if notify:
            notify(3, queue_pos)
        if queue_pos < executor_instances.free_executors():
            instance = await executor_instances.find_executor()
            task_queue.remove(task)
            if notify:
                notify(4, 0)
            result = "" #todo: implement logic
            instance.busy = False
            instance.event.set()

            if notify:
                return
            else:
                return result
        else:
            if queue_pos == 0:
                raise HTTPException(500, detail="No translator registered")
            await task_queue.wait_for_event()
@app.post("/json")
async def json(req: TranslateRequest):
    pass

@app.post("/bytes")
async def bytes(req: TranslateRequest):
    pass

@app.post("/image")
async def image(req: TranslateRequest):
    pass

@app.post("/stream_json")
async def image(req: TranslateRequest):
    pass

@app.post("/stream_bytes")
async def image(req: TranslateRequest):
    pass

@app.post("/stream_image")
async def image(req: TranslateRequest):
    pass

if __name__ == '__main__':
    import uvicorn
    from args import parse_arguments
    args = parse_arguments()
    uvicorn.run(app, host=args.host, port=args.port)