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

class Throttler:
    """
    Throttler class that throttles function calls to a specified rate, but ensures the last call is always executed.

    Example usage:

    async def my_function(arg):
        print(f'My function called with {arg}')

    throttler = Throttler(1.0)  # Throttle to 1 call per second
    throttled_function = throttler.wrap(my_function)

    async def main():
        for i in range(5):
            await throttled_function(i)
            await asyncio.sleep(0.5)  # Try to call every 0.5 seconds

        await throttler.flush()  # Execute the pending function call immediately

    asyncio.run(main())
    """
    def __init__(self, rate):
        self.rate = rate
        self.last_called = None
        self.pending_call = None
        self.pending_task = None

    def wrap(self, func):
        async def wrapped_func(*args, **kwargs):
            return await self.__call__(func, *args, **kwargs)
        return wrapped_func

    async def __call__(self, func, *args, **kwargs):
        if self.last_called:
            elapsed = asyncio.get_event_loop().time() - self.last_called
            if elapsed < self.rate:
                # If there's a pending call, cancel it
                if self.pending_call:
                    self.pending_call.cancel()
                # Schedule a new call in the future
                self.pending_task = self.__call__(func, *args, **kwargs)
                self.pending_call = asyncio.get_event_loop().call_later(
                    self.rate - elapsed,
                    asyncio.create_task,
                    self.pending_task
                )
                return

        self.last_called = asyncio.get_event_loop().time()
        self.pending_call = None
        self.pending_task = None
        return await func(*args, **kwargs)

    async def flush(self):
        if self.pending_call:
            self.pending_call.cancel()
            self.pending_call = None
            if self.pending_task:
                return await self.pending_task
