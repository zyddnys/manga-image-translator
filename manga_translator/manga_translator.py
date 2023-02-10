import asyncio
from PIL import Image
import cv2
import numpy as np
import requests
import os
from oscrypto import util as crypto_utils
import asyncio
import torch
from typing import List
import subprocess
import sys
from argparse import Namespace
import time
import traceback

from .utils import BASE_PATH, MODULE_PATH, load_image, dump_image, replace_prefix
from .args import DEFAULT_ARGS

from .detection import dispatch as dispatch_detection, prepare as prepare_detection
from .detection.ctd_utils import TextBlock
from .detection.ctd_utils.textblock import visualize_textblocks
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting
from .translators import dispatch as dispatch_translation, prepare as prepare_translation
from .text_rendering import LANGAUGE_ORIENTATION_PRESETS, dispatch as dispatch_rendering, dispatch_eng_render
from .text_rendering.text_render import count_valuable_text

# TODO: Remove legacy model checks

class MangaTranslator():

    def __init__(self, params: dict = None):
        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.verbose = params.get('verbose', False)
        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'
        self.ignore_errors = params.get('ignore_errors', False if params.get('mode', 'demo') == 'demo' else True)
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
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.abspath(path)
        dest = os.path.abspath(os.path.expanduser(dest)) if dest else ''
        params = params or {}

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
                self._report_progress('saved', True)

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
                    try:
                        img = Image.open(file_path)
                    except Exception:
                        pass
                    print('Processing', file_path, '->', output_dest)
                    output = await self.translate(img, params)
                    if output:
                        output.save(output_dest)
                        self._report_progress('saved', True)

    async def translate(self, image: Image.Image, params: dict) -> Image.Image:
        # TODO: Take list of images to speed up batch processing

        # params auto completion
        params = params or {}
        for arg in DEFAULT_ARGS:
            params.setdefault(arg, DEFAULT_ARGS[arg])
        if not 'direction' in params:
            if params['force_horizontal']:
                params['direction'] = 'h'
            elif params['force_vertical']:
                params['direction'] = 'v'
            else:
                params['direction'] = 'auto'
        params.setdefault('renderer', 'manga2eng' if params['manga2eng'] else 'default')
        # Turn dict to namespace to make values accessible through params_ns.<property>
        params_ns = Namespace(**params)

        # error checks
        if params_ns.font_path and not os.path.exists(params_ns.font_path):
            raise FileNotFoundError(params_ns.font_path)

        # preload and download (not necessary, remove to lazy load)
        print(' -- Loading models')
        await prepare_upscaling('waifu2x')
        await prepare_detection(params_ns.detector)
        await prepare_ocr(params_ns.ocr, self.device)
        await prepare_inpainting(params_ns.inpainter, self.device)
        try:
            await prepare_translation(params_ns.translator, 'auto', params_ns.target_lang)
        except Exception as e:
            self._report_progress('error', True)
            if not self.ignore_errors:
                raise e
            traceback.print_exc()
            return None

        # translate
        try:
            return await self._translate(image, params_ns)
        except Exception as e:
            self._report_progress('error', True)
            if not self.ignore_errors:
                raise e
            traceback.print_exc()
            return None

    async def _translate(self, image: Image.Image, params: Namespace) -> Image.Image:

        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if params.upscale_ratio:
            self._report_progress('upscaling')
            image = (await self._run_upscaling('waifu2x', [image], params.upscale_ratio, params.use_cuda))[0]

        img_rgb, img_alpha = load_image(image)

        # TODO: Remove once using logger tags
        print(f' -- Detector using {params.detector}')
        print(f' -- Render text direction is {params.direction}')

        self._report_progress('detection')
        text_regions, mask_raw, mask = await self._run_detection(params.detector, img_rgb, params.detection_size, params.text_threshold,
                                                                 params.box_threshold, params.unclip_ratio, params.det_rearrange_max_batches)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), mask_raw)
            bboxes = visualize_textblocks(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB), text_regions)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        if not text_regions:
            self._report_progress('no-regions', True)
            return image

        self._report_progress('ocr')
        text_regions = await self._run_ocr(params.ocr, img_rgb, text_regions)

        if not text_regions:
            self._report_progress('no-text', True)
            return image
    
        # Delayed mask refinement to take advantage of the region filtering done by ocr
        if not mask:
            self._report_progress('mask-generation')
            mask = await self._run_mask_refinement(text_regions, img_rgb, mask_raw)

        if self.verbose:
            inpaint_input_img = await self._run_inpainting('none', img_rgb, mask)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), mask)

        self._report_progress('inpainting')
        img_inpainted = await self._run_inpainting(params.inpainter, img_rgb, mask, params.inpainting_size)

        if self.verbose:
            cv2.imwrite(self._result_path('inpainted.png'), cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))

        self._report_progress('translating')
        translated_sentences = await self._run_text_translation(params.translator, 'auto', params.target_lang,
                                                                text_regions, params.mtpe)

        if not translated_sentences:
            self._report_progress('error-translating', True)
            return None

        output = await self._run_text_rendering(params.renderer, img_inpainted, text_regions, params.text_mag_ratio, params.direction,
                                                params.font_path, params.font_size_offset, img_rgb)

        self._report_progress('finished', True)
        output_image = dump_image(output, img_alpha)
        return output_image

    def _result_path(self, path: str) -> str:
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            ph(state, finished)

    def _add_logger_hook(self):
        LOG_MESSAGES = {
            'upscaling':            ' -- Running upscaling',
            'detection':            ' -- Running text detection',
            'ocr':                  ' -- Running OCR',
            'mask-generation':      ' -- Running mask refinement',
            'translating':          ' -- Translating',
            'render':               ' -- Rendering translated text',
            'saved':                ' -- Saving results',
        }
        LOG_MESSAGES_SKIP = {
            'no-regions':           ' -- No text regions! - Skipping',
            'no-text':              ' -- No text regions with text! - Skipping',
        }
        LOG_MESSAGES_ERROR = {
            'error-translating':    ' -- ERROR Text translator returned empty queries',
        }
        def ph(state, finished):
            if state in LOG_MESSAGES:
                print(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                print(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_ERROR:
                print(LOG_MESSAGES[state])

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

        translated = await dispatch_translation(key, src_lang, tgt_lang, [r.get_text() for r in text_regions], use_mtpe,
                                                'cpu' if self._cuda_limited_memory else self.device)
        for blk, tr in zip(text_regions, translated):
            blk.translation = tr
            blk.target_lang = tgt_lang
        return translated

    async def _run_text_rendering(self, key: str, img: np.ndarray, text_regions: List[TextBlock], text_mag_ratio: np.integer, text_direction: str,
                                 font_path: str = '', font_size_offset: int = 0, original_img: np.ndarray = None, mask: np.ndarray = None):

        # manga2eng currently only supports horizontal rendering
        if key == 'manga2eng' and text_regions and LANGAUGE_ORIENTATION_PRESETS.get[text_regions[0].target_lang] == 'h':
            output = await dispatch_eng_render(img, original_img, text_regions, font_path)
        else:
            output = await dispatch_rendering(img, text_regions, text_mag_ratio, text_direction, font_path, font_size_offset, original_img)
        return output

import atexit
import signal

class MangaTranslatorWeb(MangaTranslator):
    """
    Translator client that executes tasks on behalf of the server in web_main.py.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = str(params.get('port', '5003'))
        self.nonce = params.get('nonce', self.generate_nonce())
        self.log_web = params.get('log_web', False)
        self.ignore_errors = params.get('ignore_errors', True)
        print(params.get('ignore_errors'), self.ignore_errors)
        self._task_id = None
        self._params = None

    def generate_nonce(self):
        return crypto_utils.rand_bytes(16).hex()

    def instantiate_webserver(self):
        os.setpgrp()
        web_executable = [sys.executable, '-u'] if self.log_web else [sys.executable]
        web_process_args = [os.path.join(MODULE_PATH, 'web_main.py'), self.nonce, self.host, self.port]
        extra_web_args = {'stdout': sys.stdout, 'stderr': sys.stderr} if self.log_web else {}
        subprocess.Popen([*web_executable, *web_process_args], **extra_web_args)
        # https://stackoverflow.com/a/322317
        atexit.register(self.terminate_webserver)

    def terminate_webserver(self):
        os.killpg(0, signal.SIGKILL)

    async def listen(self, translation_params: dict = None):
        """
        Listens for translation tasks from web server.
        """
        print(' -- Running in web service mode')
        print(' -- Waiting for translation tasks')

        def sync_state(state: str, finished: bool):
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
            print(f' -- Processing task {self._task_id}')
            if translation_params:
                for p, value in translation_params.items():
                    self._params.setdefault(p, value)
            await self.translate_path(self._result_path('input.png'), self._result_path('final.png'), params=self._params)

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
                    return translated
                await asyncio.sleep(0.1)
        else:
            return await super()._run_text_translation(key, src_lang, tgt_lang, text_regions, use_mtpe)

class MangaTranslatorWS(MangaTranslator):
    async def _run_text_rendering(self, key: str, img: np.ndarray, text_mag_ratio: np.integer, text_regions: List[TextBlock], text_direction: str,
                                 font_path: str = '', font_size_offset: int = 0, original_img: np.ndarray = None, mask: np.ndarray = None):

        img_inpainted = np.copy(img)
        render_mask = np.copy(mask)
        render_mask[render_mask < 127] = 0
        render_mask[render_mask >= 127] = 1
        render_mask = render_mask[:, :, None]

        output = await super()._run_text_rendering(key, img, text_mag_ratio, text_regions, text_direction, font_path, font_size_offset, original_img, mask)

        # only keep sections in mask
        if self.verbose:
            cv2.imwrite(f'result/ws_inmask.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGRA) * render_mask)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA) * render_mask

        return output
