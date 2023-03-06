import asyncio
from PIL import Image
import cv2
import numpy as np
import requests
import os
from oscrypto import util as crypto_utils
import torch
from typing import List
import subprocess
import sys
import time
import atexit
import logging

from .args import DEFAULT_ARGS
from .utils import (
    BASE_PATH,
    MODULE_PATH,
    LANGAUGE_ORIENTATION_PRESETS,
    ModelWrapper,
    TextBlock,
    Context,
    load_image,
    dump_image,
    replace_prefix,
    visualize_textblocks,
    add_file_logger,
    remove_file_logger,
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting
from .translators import LanguageUnsupportedException, dispatch as dispatch_translation, prepare as prepare_translation
from .text_rendering import dispatch as dispatch_rendering, dispatch_eng_render
from .text_rendering.text_render import count_valuable_text

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

def set_main_logger(l):
    global logger
    logger = l

class MangaTranslator():

    def __init__(self, params: dict = None):
        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.verbose = params.get('verbose', False)
        self.ignore_errors = params.get('ignore_errors', False if params.get('mode', 'demo') == 'demo' else True)

        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'
        self._cuda_limited_memory = params.get('use_cuda_limited', False)
        if self._cuda_limited_memory and not self.using_cuda:
            self.device = 'cuda'
        if self.using_cuda and not torch.cuda.is_available():
            raise Exception('CUDA compatible device could not be found whilst --use-cuda args was set...')

        self.result_sub_folder = ''

    @property
    def using_cuda(self):
        return self.device.startswith('cuda')

    async def translate_path(self, path: str, dest: str = None, params: dict = None):
        """
        Translates an image or folder (recursively) specified through the path.
        """
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.abspath(path)
        dest = os.path.abspath(os.path.expanduser(dest)) if dest else ''
        params = params or {}

        # TODO: accept * in file paths

        if os.path.isfile(path):
            # Determine destination file path
            if not dest:
                # Use the same folder as the source
                p, ext = os.path.splitext(path)
                dest = f'{p}-translated{ext}'
            elif not os.path.basename(dest):
                # If the folders differ use the original filename from the source
                if os.path.dirname(path) != dest:
                    dest = os.path.join(dest, os.path.basename(path))
                else:
                    p, ext = os.path.splitext(os.path.basename(path))
                    dest = os.path.join(dest, f'{p}-translated{ext}')
            dest_root = os.path.dirname(dest)

            output = await self.translate(Image.open(path), params)
            if output:
                os.makedirs(dest_root, exist_ok=True)
                output.save(dest)
                await self._report_progress('saved', True)

        elif os.path.isdir(path):
            # Determine destination folder path
            if path[-1] == '\\' or path[-1] == '/':
                path = path[:-1]
            dest = dest or path + '-translated'
            if os.path.exists(dest) and not os.path.isdir(dest):
                raise FileExistsError(dest)

            for root, subdirs, files in os.walk(path):
                dest_root = replace_prefix(root, path, dest)
                os.makedirs(dest_root, exist_ok=True)
                for f in files:
                    if f.lower() == '.thumb':
                        continue
                    file_path = os.path.join(root, f)
                    output_dest = replace_prefix(file_path, path, dest)
                    if os.path.exists(output_dest):
                        continue
                    img = None
                    try:
                        img = Image.open(file_path)
                    except Exception:
                        pass
                    if img:
                        print()
                        logger.info(f'Processing {file_path} -> {output_dest}')
                        output = await self.translate(img, params)
                        if output:
                            output.save(output_dest)
                            await self._report_progress('saved', True)

    async def translate(self, image: Image.Image, params: dict = None) -> Image.Image:
        """
        Translates a PIL image preferably from a manga.
        Returns `None` if an error occured and `image` if no text was found.
        """
        # TODO: Take list of images to speed up batch processing

        # Turn dict to context to make values accessible through params.<property>
        params = params or {}
        params = Context(**params)

        # params auto completion
        for arg in DEFAULT_ARGS:
            params.setdefault(arg, DEFAULT_ARGS[arg])
        if 'direction' not in params:
            if params.force_horizontal:
                params.direction = 'h'
            elif params.force_vertical:
                params.direction = 'v'
            else:
                params.direction = 'auto'
        if 'alignment' not in params:
            if params.align_left:
                params.alignment = 'left'
            elif params.align_center:
                params.alignment = 'center'
            elif params.align_right:
                params.alignment = 'right'
            else:
                params.alignment = 'auto'
        params.setdefault('renderer', 'manga2eng' if params['manga2eng'] else 'default')

        try:
            # preload and download models (not necessary, remove to lazy load)
            logger.info('Loading models')
            if params.model_dir:
                ModelWrapper._MODEL_DIR = params.model_dir
            if params.upscale_ratio:
                await prepare_upscaling(params.upscaler)
            await prepare_detection(params.detector)
            await prepare_ocr(params.ocr, self.device)
            await prepare_inpainting(params.inpainter, self.device)
            await prepare_translation(params.translator, 'auto', params.target_lang)

            # translate
            return await self._translate(image, params)
        except Exception as e:
            if isinstance(e, LanguageUnsupportedException):
                await self._report_progress('error-lang', True)
            else:
                await self._report_progress('error', True)
            if not self.ignore_errors:
                raise
            else:
                logger.error(f'{e.__class__.__name__}: {e}',
                             exc_info=e if self.verbose else None)
            return None

    async def _translate(self, image: Image.Image, params: Context) -> Image.Image:
        # TODO: Split up into self sufficient functions that call what they need automatically

        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if params.upscale_ratio:
            await self._report_progress('upscaling')
            image_upscaled = (await self._run_upscaling(params.upscaler, [image], params.upscale_ratio))[0]
        else:
            image_upscaled = image

        img_rgb, img_alpha = load_image(image_upscaled)

        await self._report_progress('detection')
        text_regions, mask_raw, mask = await self._run_detection(params.detector, img_rgb, params.detection_size, params.text_threshold,
                                                                 params.box_threshold, params.unclip_ratio, params.det_rearrange_max_batches)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), mask_raw)
            bboxes = visualize_textblocks(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), text_regions)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        if not text_regions:
            await self._report_progress('skip-no-regions', True)
            return image

        await self._report_progress('ocr')
        text_regions = await self._run_ocr(params.ocr, img_rgb, text_regions)

        if not text_regions:
            await self._report_progress('skip-no-text', True)
            return image

        # Delayed mask refinement to take advantage of the region filtering done by ocr
        if mask is None:
            await self._report_progress('mask-generation')
            mask = await self._run_mask_refinement(text_regions, img_rgb, mask_raw)

        if self.verbose:
            inpaint_input_img = await self._run_inpainting('none', img_rgb, mask)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), mask)

        await self._report_progress('inpainting')
        img_inpainted = await self._run_inpainting(params.inpainter, img_rgb, mask, params.inpainting_size)

        if self.verbose:
            cv2.imwrite(self._result_path('inpainted.png'), cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))

        await self._report_progress('translating')
        translated_sentences = await self._run_text_translation(params.translator, 'auto', params.target_lang,
                                                                text_regions, params.mtpe)

        if not translated_sentences:
            await self._report_progress('error-translating', True)
            return None

        await self._report_progress('rendering')
        for i, region in enumerate(text_regions):
            region.translation = translated_sentences[i]
            region.target_lang = params.target_lang
            region._alignment = params.alignment
            region._direction = params.direction

        output = await self._run_text_rendering(params.renderer, img_inpainted, text_regions, params.text_mag_ratio, params.direction,
                                                params.font_path, params.font_size_offset, params.font_size_minimum, img_rgb,
                                                mask, rearrange_regions=(params.inpainter != 'none'))

        if params.downscale:
            await self._report_progress('downscaling')
            if img_alpha:
                # Add alpha channel to rgb
                output = np.concatenate([output.astype(np.uint8), np.array(img_alpha).astype(np.uint8)[..., None]], axis=2)
            output = cv2.resize(output, image.size, interpolation=cv2.INTER_LINEAR)

        await self._report_progress('finished', True)
        output_image = dump_image(output, img_alpha)

        return output_image

    def _result_path(self, path: str) -> str:
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        LOG_MESSAGES = {
            'upscaling':            'Running upscaling',
            'detection':            'Running text detection',
            'ocr':                  'Running OCR',
            'mask-generation':      'Running mask refinement',
            'translating':          'Translating',
            'rendering':            'Rendering translated text',
            'downscaling':          'Running downscaling',
            'saved':                'Saving results',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions':      'No text regions! - Skipping',
            'skip-no-text':         'No text regions with text! - Skipping',
        }
        LOG_MESSAGES_ERROR = {
            'error-translating':    'Text translator returned empty queries',
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    # TODO: Maybe find a better way to wrap the dispatch functions to reduce redundancy (decorators? dicts?)

    async def _run_upscaling(self, key: str, image_batch: List[Image.Image], upscale_ratio: int):
        return await dispatch_upscaling(key, image_batch, upscale_ratio, self.device)

    async def _run_detection(self, key: str, img: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                             unclip_ratio: float, det_rearrange_max_batches: int):
        return await dispatch_detection(key, img, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches,
                                        self.device, self.verbose)

    async def _run_ocr(self, key: str, img: np.ndarray, text_regions: List[TextBlock]):
        text_regions = await dispatch_ocr(key, img, text_regions, self.device, self.verbose)

        # Filter regions by their text
        text_regions = list(filter(lambda r: count_valuable_text(r.get_text()) > 1 and not r.get_text().isnumeric(), text_regions))
        return text_regions

    async def _run_mask_refinement(self, text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text'):
        return await dispatch_mask_refinement(text_regions, raw_image, raw_mask, method, self.verbose)

    async def _run_inpainting(self, key: str, img: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024):
        return await dispatch_inpainting(key, img, mask, inpainting_size, self.using_cuda, self.verbose)

    async def _run_text_translation(self, key: str, src_lang: str, tgt_lang: str, text_regions: List[TextBlock], use_mtpe: bool = False):
        return await dispatch_translation(key, src_lang, tgt_lang, [r.get_text() for r in text_regions], use_mtpe,
                                                'cpu' if self._cuda_limited_memory else self.device)

    async def _run_text_rendering(self, key: str, img: np.ndarray, text_regions: List[TextBlock], text_mag_ratio: np.integer,
                                  text_direction: str = 'auto', font_path: str = '', font_size_offset: int = 0, font_size_minimum: int = 0,
                                  original_img: np.ndarray = None, mask: np.ndarray = None, rearrange_regions: bool = False):
        # manga2eng currently only supports horizontal rendering
        if key == 'manga2eng' and text_regions and LANGAUGE_ORIENTATION_PRESETS.get(text_regions[0].target_lang) == 'h':
            output = await dispatch_eng_render(img, original_img, text_regions, font_path)
        else:
            output = await dispatch_rendering(img, text_regions, text_mag_ratio, font_path, font_size_offset, font_size_minimum, rearrange_regions, mask)
        return output


class MangaTranslatorWeb(MangaTranslator):
    """
    Translator client that executes tasks on behalf of the webserver in web_main.py.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = str(params.get('port', '5003'))
        self.nonce = params.get('nonce', None)
        if not isinstance(self.nonce, str):
            self.nonce = self.generate_nonce()
        self.log_web = params.get('log_web', False)
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None

    def generate_nonce(self):
        return crypto_utils.rand_bytes(16).hex()

    def instantiate_webserver(self):
        web_executable = [sys.executable, '-u'] if self.log_web else [sys.executable]
        web_process_args = [os.path.join(MODULE_PATH, 'server', 'web_main.py'), self.nonce, self.host, str(self.port)]
        extra_web_args = {'stdout': sys.stdout, 'stderr': sys.stderr} if self.log_web else {}
        proc = subprocess.Popen([*web_executable, *web_process_args], **extra_web_args)
        atexit.register(proc.terminate)

    async def listen(self, translation_params: dict = None):
        """
        Listens for translation tasks from web server.
        """
        logger.info('Waiting for translation tasks')

        async def sync_state(state: str, finished: bool):
            # wait for translation to be saved first (bad solution?)
            finished = finished and not state == 'finished'
            while True:
                try:
                    data = {
                        'task_id': self._task_id,
                        'nonce': self.nonce,
                        'state': state,
                        'finished': finished,
                    }
                    requests.post(f'http://{self.host}:{self.port}/task-update-internal', json=data, timeout=20)
                    break
                except Exception:
                    # if translation is finished server has to know
                    if finished:
                        continue
                    else:
                        break
        self.add_progress_hook(sync_state)

        while True:
            self._task_id, self._params = self._get_task()
            if self._params and 'exit' in self._params:
                break
            if not (self._task_id and self._params):
                await asyncio.sleep(0.1)
                continue

            self.result_sub_folder = self._task_id
            logger.info(f'Processing task {self._task_id}')
            if translation_params is not None:
                # Combine default params with params chosen by webserver
                for p, default_value in translation_params.items():
                    current_value = self._params.get(p)
                    self._params[p] = current_value if current_value is not None else default_value
            if self.verbose:
                # Write log file
                log_file = self._result_path('log.txt')
                add_file_logger(log_file)

            await self.translate_path(self._result_path('input.png'), self._result_path('final.png'), params=self._params)

            if self.verbose:
                remove_file_logger(log_file)
            self._task_id = None
            self._params = None
            self.result_sub_folder = ''

    def _get_task(self):
        try:
            rjson = requests.get(f'http://{self.host}:{self.port}/task-internal?nonce={self.nonce}', timeout=3600).json()
            return rjson.get('task_id'), rjson.get('data')
        except Exception:
            return None, None

    async def _run_ocr(self, key: str, img: np.ndarray, regions: List[TextBlock]):
        regions = await super()._run_ocr(key, img, regions)
        if self._params.get('manual', False):
            requests.post(f'http://{self.host}:{self.port}/request-translation-internal', json={
                'task_id': self._task_id,
                'nonce': self.nonce,
                'texts': [r.get_text() for r in regions],
            }, timeout=20)
        return regions

    async def _run_text_translation(self, key: str, src_lang: str, tgt_lang: str, text_regions: List[TextBlock], use_mtpe: bool = False):
        if self._params.get('manual', False):
            requests.post(f'http://{self.host}:{self.port}/request-translation-internal', json={
                'task_id': self._task_id,
                'nonce': self.nonce,
                'texts': [r.get_text() for r in text_regions]
            }, timeout=20)

            # wait for at most 1 hour for manual translation
            wait_until = time.time() + 3600
            while time.time() < wait_until:
                ret = requests.post(f'http://{self.host}:{self.port}/get-translation-result-internal', json={
                    'task_id': self._task_id,
                    'nonce': self.nonce
                }, timeout=20).json()
                if 'result' in ret:
                    translated = ret['result']
                    if isinstance(translated, str):
                        if translated == 'error':
                            return None
                    for blk, tr in zip(text_regions, translated):
                        blk.translation = tr
                        blk.target_lang = tgt_lang
                    return translated
                await asyncio.sleep(0.1)
        else:
            return await super()._run_text_translation(key, src_lang, tgt_lang, text_regions, use_mtpe)


class MangaTranslatorWS(MangaTranslator):

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.url = params.get('ws_url')
        self.secret = params.get('ws_secret', os.getenv('WS_SECRET', ''))
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None

    async def listen(self, translation_params: dict = None):
        import io
        import shutil
        import websockets
        import manga_translator.server.ws_pb2 as ws_pb2

        async for websocket in websockets.connect(self.url, extra_headers={'x-secret': self.secret}, max_size=100_000_000):
            try:
                logger.info('Connected to websocket server')

                async def sync_state(state, finished):
                    msg = ws_pb2.WebSocketMessage()
                    msg.status.id = self._task_id
                    msg.status.status = state
                    await websocket.send(msg.SerializeToString())

                self.add_progress_hook(sync_state)

                async for raw in websocket:
                    msg = ws_pb2.WebSocketMessage()
                    msg.ParseFromString(raw)
                    if msg.WhichOneof('message') == 'new_task':
                        task = msg.new_task
                        self._task_id = task.id

                        if self.verbose:
                            shutil.rmtree(f'result/{self._task_id}', ignore_errors=True)
                            os.makedirs(f'result/{self._task_id}', exist_ok=True)

                        params = {
                            'target_language': task.target_language,
                            'detector': task.detector,
                            'direction': task.direction,
                            'translator': task.translator,
                            'size': task.size,
                        }

                        logger.info(f'-- Processing task {self._task_id}')
                        if translation_params:
                            for p, default_value in translation_params.items():
                                current_value = params.get(p)
                                params[p] = current_value if current_value is not None else default_value
                        image = Image.open(io.BytesIO(task.source_image))
                        output = await self.translate(image, params)
                        if output:
                            img = io.BytesIO()
                            if output == image:
                                output = Image.fromarray(np.zeros((output.height, output.width, 4), dtype=np.uint8))
                            output.save(img, format='PNG')
                            if self.verbose:
                                output.save(self._result_path('ws_final.png'))

                            img_bytes = img.getvalue()

                            result = ws_pb2.WebSocketMessage()
                            result.finish_task.id = self._task_id
                            result.finish_task.translation_mask = img_bytes
                            await websocket.send(result.SerializeToString())

                        logger.info('Waiting for translation tasks')
                        self._task_id = None

            except Exception as e:
                logger.error(f'{e.__class__.__name__}: {e}', exc_info=e if self.verbose else None)

    async def _run_text_rendering(self, key: str, img: np.ndarray, text_regions: List[TextBlock], text_mag_ratio: np.integer,
                                  text_direction: str = 'auto', font_path: str = '', font_size_offset: int = 0, font_size_minimum: int = 0,
                                  original_img: np.ndarray = None, mask: np.ndarray = None, rearrange_regions: bool = False):

        img_inpainted = np.copy(img)
        render_mask = np.copy(mask)
        render_mask[render_mask < 127] = 0
        render_mask[render_mask >= 127] = 1
        render_mask = render_mask[:, :, None]

        output = await super()._run_text_rendering(key, img, text_mag_ratio, text_regions, text_direction, font_path, font_size_offset,
                                                   font_size_minimum, original_img, render_mask, rearrange_regions)
        render_mask[np.sum(img != output, axis=2) > 0] = 1
        if self.verbose:
            cv2.imwrite(self._result_path('ws_render_in.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_render_out.png'), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_mask.png'), render_mask * 255)

        # only keep sections in mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_inmask.png'), cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGRA) * render_mask)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA) * render_mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_output.png'), cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA) * render_mask)

        return output
