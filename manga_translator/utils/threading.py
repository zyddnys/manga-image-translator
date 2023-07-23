import asyncio

class PriorityLock:
    """
    Lock object which prioritizes each acquire
    License: MIT <tuxtimo@gmail.com>
    """
    class _Context:
        def __init__(self, lock: 'PriorityLock', priority: int):
            self._lock = lock
            self._priority = priority

        async def __aenter__(self):
            await self._lock.acquire(self._priority)

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._lock.release()

    def __init__(self):
        self._lock = asyncio.Lock()
        self._acquire_queue = asyncio.PriorityQueue()
        self._need_to_wait = False

    async def acquire(self, priority: int):
        async with self._lock:
            if not self._need_to_wait:
                self._need_to_wait = True
                return True

            event = asyncio.Event()
            await self._acquire_queue.put((priority, event))
        await event.wait()
        return True

    async def release(self):
        async with self._lock:
            try:
                event: asyncio.Event
                _, event = self._acquire_queue.get_nowait()
            except asyncio.QueueEmpty:
                self._need_to_wait = False
            else:
                event.set()

    def __call__(self, priority: int):
        return self._Context(self, priority)
