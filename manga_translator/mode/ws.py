import asyncio
import logging
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from manga_translator import logger, Context, MangaTranslator, Config
from manga_translator.utils import PriorityLock, Throttler


class MangaTranslatorWS(MangaTranslator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.url = params.get('ws_url')
        self.secret = params.get('ws_secret', os.getenv('WS_SECRET', ''))
        self.ignore_errors = params.get('ignore_errors', True)

        self._task_id = None
        self._websocket = None

    async def listen(self, translation_params: dict = None):
        from threading import Thread
        import io
        import aioshutil
        from aiofiles import os
        import websockets
        from ..server import ws_pb2

        self._server_loop = asyncio.new_event_loop()
        self.task_lock = PriorityLock()
        self.counter = 0

        async def _send_and_yield(websocket, msg):
            # send message and yield control to the event loop (to actually send the message)
            await websocket.send(msg)
            await asyncio.sleep(0)

        send_throttler = Throttler(0.2)
        send_and_yield = send_throttler.wrap(_send_and_yield)

        async def sync_state(state, finished):
            if self._websocket is None:
                return
            msg = ws_pb2.WebSocketMessage()
            msg.status.id = self._task_id
            msg.status.status = state
            self._server_loop.call_soon_threadsafe(
                asyncio.create_task,
                send_and_yield(self._websocket, msg.SerializeToString())
            )

        self.add_progress_hook(sync_state)

        async def translate(task_id, websocket, image, params):
            async with self.task_lock((1 << 31) - params['ws_count']):
                self._task_id = task_id
                self._websocket = websocket
                result = await self.translate(image, params)
                self._task_id = None
                self._websocket = None
            return result

        async def server_send_status(websocket, task_id, status):
            msg = ws_pb2.WebSocketMessage()
            msg.status.id = task_id
            msg.status.status = status
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0)

        async def server_process_inner(main_loop, logger_task, session, websocket, task) -> Tuple[bool, bool]:
            logger_task.info(f'-- Processing task {task.id}')
            await server_send_status(websocket, task.id, 'pending')

            if self.verbose:
                await aioshutil.rmtree(f'result/{task.id}', ignore_errors=True)
                await os.makedirs(f'result/{task.id}', exist_ok=True)

            params = {
                'target_lang': task.target_language,
                'skip_lang': task.skip_language,
                'detector': task.detector,
                'direction': task.direction,
                'translator': task.translator,
                'size': task.size,
                'ws_event_loop': asyncio.get_event_loop(),
                'ws_count': self.counter,
            }
            self.counter += 1

            logger_task.info(f'-- Downloading image from {task.source_image}')
            await server_send_status(websocket, task.id, 'downloading')
            async with session.get(task.source_image) as resp:
                if resp.status == 200:
                    source_image = await resp.read()
                else:
                    msg = ws_pb2.WebSocketMessage()
                    msg.status.id = task.id
                    msg.status.status = 'error-download'
                    await websocket.send(msg.SerializeToString())
                    await asyncio.sleep(0)
                    return False, False

            logger_task.info(f'-- Translating image')
            if translation_params:
                for p, default_value in translation_params.items():
                    current_value = params.get(p)
                    params[p] = current_value if current_value is not None else default_value

            image = Image.open(io.BytesIO(source_image))

            (ori_w, ori_h) = image.size
            if max(ori_h, ori_w) > 1200:
                params['upscale_ratio'] = 1

            await server_send_status(websocket, task.id, 'preparing')
            # translation_dict = await self.translate(image, params)
            translation_dict = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    translate(task.id, websocket, image, params),
                    main_loop
                )
            )
            await send_throttler.flush()

            output: Image.Image = translation_dict.result
            if output is not None:
                await server_send_status(websocket, task.id, 'saving')

                output = output.resize((ori_w, ori_h), resample=Image.LANCZOS)

                img = io.BytesIO()
                output.save(img, format='PNG')
                if self.verbose:
                    output.save(self._result_path('ws_final.png'))

                img_bytes = img.getvalue()
                logger_task.info(f'-- Uploading result to {task.translation_mask}')
                await server_send_status(websocket, task.id, 'uploading')
                async with session.put(task.translation_mask, data=img_bytes) as resp:
                    if resp.status != 200:
                        logger_task.error(f'-- Failed to upload result:')
                        logger_task.error(f'{resp.status}: {resp.reason}')
                        msg = ws_pb2.WebSocketMessage()
                        msg.status.id = task.id
                        msg.status.status = 'error-upload'
                        await websocket.send(msg.SerializeToString())
                        await asyncio.sleep(0)
                        return False, False

            return True, output is not None

        async def server_process(main_loop, session, websocket, task):
            logger_task = logger.getChild(f'{task.id}')
            try:
                (success, has_translation_mask) = await server_process_inner(main_loop, logger_task, session, websocket,
                                                                             task)
            except Exception as e:
                logger_task.error(f'-- Task failed with exception:')
                logger_task.error(f'{e.__class__.__name__}: {e}', exc_info=e if self.verbose else None)
                (success, has_translation_mask) = False, False
            finally:
                result = ws_pb2.WebSocketMessage()
                result.finish_task.id = task.id
                result.finish_task.success = success
                result.finish_task.has_translation_mask = has_translation_mask
                await websocket.send(result.SerializeToString())
                await asyncio.sleep(0)
                logger_task.info(f'-- Task finished')

        async def async_server_thread(main_loop):
            from aiohttp import ClientSession, ClientTimeout
            timeout = ClientTimeout(total=30)
            async with ClientSession(timeout=timeout) as session:
                logger_conn = logger.getChild('connection')
                if self.verbose:
                    logger_conn.setLevel(logging.DEBUG)
                async for websocket in websockets.connect(
                        self.url,
                        extra_headers={
                            'x-secret': self.secret,
                        },
                        max_size=1_000_000,
                        logger=logger_conn
                ):
                    bg_tasks = set()
                    try:
                        logger.info('-- Connected to websocket server')

                        async for raw in websocket:
                            # logger.info(f'Got message: {raw}')
                            msg = ws_pb2.WebSocketMessage()
                            msg.ParseFromString(raw)
                            if msg.WhichOneof('message') == 'new_task':
                                task = msg.new_task
                                bg_task = asyncio.create_task(server_process(main_loop, session, websocket, task))
                                bg_tasks.add(bg_task)
                                bg_task.add_done_callback(bg_tasks.discard)

                    except Exception as e:
                        logger.error(f'{e.__class__.__name__}: {e}', exc_info=e if self.verbose else None)

                    finally:
                        logger.info('-- Disconnected from websocket server')
                        for bg_task in bg_tasks:
                            bg_task.cancel()

        def server_thread(future, main_loop, server_loop):
            asyncio.set_event_loop(server_loop)
            try:
                server_loop.run_until_complete(async_server_thread(main_loop))
            finally:
                future.set_result(None)

        future = asyncio.Future()
        Thread(
            target=server_thread,
            args=(future, asyncio.get_running_loop(), self._server_loop),
            daemon=True
        ).start()

        # create a future that is never done
        await future

    async def _run_text_translation(self, config: Config, ctx: Context):
        coroutine = super()._run_text_translation(config, ctx)
        if config.translator.translator_gen.has_offline():
            return await coroutine
        else:
            task_id = self._task_id
            websocket = self._websocket
            await self.task_lock.release()
            result = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    coroutine,
                    ctx.ws_event_loop
                )
            )
            await self.task_lock.acquire((1 << 30) - ctx.ws_count)
            self._task_id = task_id
            self._websocket = websocket
            return result

    async def _run_text_rendering(self, config: Config, ctx: Context):
        render_mask = (ctx.mask >= 127).astype(np.uint8)[:, :, None]

        output = await super()._run_text_rendering(config, ctx)
        render_mask[np.sum(ctx.img_rgb != output, axis=2) > 0] = 1
        ctx.render_mask = render_mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_render_in.png'), cv2.cvtColor(ctx.img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_render_out.png'), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_mask.png'), render_mask * 255)

        # only keep sections in mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_inmask.png'), cv2.cvtColor(ctx.img_rgb, cv2.COLOR_RGB2BGRA) * render_mask)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA) * render_mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_output.png'), cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA) * render_mask)

        return output
