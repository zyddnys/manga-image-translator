# Experimental. May be replaced by a refactored server/web_main.py in the future.
import asyncio
import base64
import io

import cv2
import numpy as np
from PIL import Image
from aiohttp import web
from aiohttp.web_middlewares import middleware
from marshmallow import fields, Schema, ValidationError

from manga_translator import MangaTranslator, Context, TranslationInterrupt, logger
from manga_translator.args import translator_chain
from manga_translator.detection import DETECTORS
from manga_translator.inpainting import INPAINTERS
from manga_translator.manga_translator import _preprocess_params
from manga_translator.ocr import OCRS
from manga_translator.translators import VALID_LANGUAGES, TRANSLATORS
from manga_translator.upscaling import UPSCALERS


class MangaTranslatorAPI(MangaTranslator):
    def __init__(self, params: dict = None):
        import nest_asyncio
        nest_asyncio.apply()
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = params.get('port', '5003')
        self.log_web = params.get('log_web', False)
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None
        self.params = params
        self.queue = []

    async def wait_queue(self, id: int):
        while self.queue[0] != id:
            await asyncio.sleep(0.05)

    def remove_from_queue(self, id: int):
        self.queue.remove(id)

    def generate_id(self):
        try:
            x = max(self.queue)
        except:
            x = 0
        return x + 1

    def middleware_factory(self):
        @middleware
        async def sample_middleware(request, handler):
            id = self.generate_id()
            self.queue.append(id)
            try:
                await self.wait_queue(id)
            except Exception as e:
                print(e)
            try:
                # todo make cancellable
                response = await handler(request)
            except:
                response = web.json_response({'error': "Internal Server Error", 'status': 500},
                                             status=500)
            # Handle cases where a user leaves the queue, request fails, or is completed
            try:
                self.remove_from_queue(id)
            except Exception as e:
                print(e)
            return response

        return sample_middleware

    async def get_file(self, image, base64Images, url) -> Image:
        if image is not None:
            content = image.file.read()
        elif base64Images is not None:
            base64Images = base64Images
            if base64Images.__contains__('base64,'):
                base64Images = base64Images.split('base64,')[1]
            content = base64.b64decode(base64Images)
        elif url is not None:
            from aiohttp import ClientSession
            async with ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                    else:
                        return web.json_response({'status': 'error'})
        else:
            raise ValidationError("donest exist")
        img = Image.open(io.BytesIO(content))

        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width * img.height > 8000 ** 2:
            raise ValidationError("to large")
        return img

    async def listen(self, translation_params: dict = None):
        self.params = translation_params
        app = web.Application(client_max_size=1024 * 1024 * 50, middlewares=[self.middleware_factory()])

        routes = web.RouteTableDef()
        run_until_state = ''

        async def hook(state, finished):
            if run_until_state and run_until_state == state and not finished:
                raise TranslationInterrupt()

        self.add_progress_hook(hook)

        @routes.post("/get_text")
        async def text_api(req):
            nonlocal run_until_state
            run_until_state = 'translating'
            return await self.err_handling(self.run_translate, req, self.format_translate)

        @routes.post("/translate")
        async def translate_api(req):
            nonlocal run_until_state
            run_until_state = 'after-translating'
            return await self.err_handling(self.run_translate, req, self.format_translate)

        @routes.post("/inpaint_translate")
        async def inpaint_translate_api(req):
            nonlocal run_until_state
            run_until_state = 'rendering'
            return await self.err_handling(self.run_translate, req, self.format_translate)

        @routes.post("/colorize_translate")
        async def colorize_translate_api(req):
            nonlocal run_until_state
            run_until_state = 'rendering'
            return await self.err_handling(self.run_translate, req, self.format_translate, True)

        # #@routes.post("/file")
        # async def file_api(req):
        #     #TODO: return file
        #     return await self.err_handling(self.file_exec, req, None)

        app.add_routes(routes)
        web.run_app(app, host=self.host, port=self.port)

    async def run_translate(self, translation_params, img):
        return await self.translate(img, translation_params)

    async def err_handling(self, func, req, format, ri=False):
        try:
            if req.content_type == 'application/json' or req.content_type == 'multipart/form-data':
                if req.content_type == 'application/json':
                    d = await req.json()
                else:
                    d = await req.post()
                schema = self.PostSchema()
                data = schema.load(d)
                if 'translator_chain' in data:
                    data['translator_chain'] = translator_chain(data['translator_chain'])
                if 'selective_translation' in data:
                    data['selective_translation'] = translator_chain(data['selective_translation'])
                ctx = Context(**dict(self.params, **data))
                _preprocess_params(ctx)
                if data.get('image') is None and data.get('base64Images') is None and data.get('url') is None:
                    return web.json_response({'error': "Missing input", 'status': 422})
                fil = await self.get_file(data.get('image'), data.get('base64Images'), data.get('url'))
                if 'image' in data:
                    del data['image']
                if 'base64Images' in data:
                    del data['base64Images']
                if 'url' in data:
                    del data['url']
                attempts = 0
                while ctx.attempts == -1 or attempts <= ctx.attempts:
                    if attempts > 0:
                        logger.info(f'Retrying translation! Attempt {attempts}' + (
                            f' of {ctx.attempts}' if ctx.attempts != -1 else ''))
                    try:
                        await func(ctx, fil)
                        break
                    except TranslationInterrupt:
                        break
                    except Exception as e:
                        print(e)
                    attempts += 1
                if ctx.attempts != -1 and attempts > ctx.attempts:
                    return web.json_response({'error': "Internal Server Error", 'status': 500},
                                             status=500)
                try:
                    return format(ctx, ri)
                except Exception as e:
                    print(e)
                    return web.json_response({'error': "Failed to format", 'status': 500},
                                             status=500)
            else:
                return web.json_response({'error': "Wrong content type: " + req.content_type, 'status': 415},
                                         status=415)
        except ValueError as e:
            print(e)
            return web.json_response({'error': "Wrong input type", 'status': 422}, status=422)

        except ValidationError as e:
            print(e)
            return web.json_response({'error': "Input invalid", 'status': 422}, status=422)

    def format_translate(self, ctx: Context, return_image: bool):
        text_regions = ctx.text_regions
        inpaint = ctx.img_inpainted
        results = []
        if 'overlay_ext' in ctx:
            overlay_ext = ctx['overlay_ext']
        else:
            overlay_ext = 'jpg'
        for i, blk in enumerate(text_regions):
            minX, minY, maxX, maxY = blk.xyxy
            if 'translations' in ctx:
                trans = {key: value[i] for key, value in ctx['translations'].items()}
            else:
                trans = {}
            trans["originalText"] = text_regions[i].text
            if inpaint is not None:
                overlay = inpaint[minY:maxY, minX:maxX]

                retval, buffer = cv2.imencode('.' + overlay_ext, overlay)
                jpg_as_text = base64.b64encode(buffer)
                background = "data:image/" + overlay_ext + ";base64," + jpg_as_text.decode("utf-8")
            else:
                background = None
            text_region = text_regions[i]
            text_region.adjust_bg_color = False
            color1, color2 = text_region.get_font_colors()

            results.append({
                'text': trans,
                'minX': int(minX),
                'minY': int(minY),
                'maxX': int(maxX),
                'maxY': int(maxY),
                'textColor': {
                    'fg': color1.tolist(),
                    'bg': color2.tolist()
                },
                'language': text_regions[i].source_lang,
                'background': background
            })
        if return_image and ctx.img_colorized is not None:
            retval, buffer = cv2.imencode('.' + overlay_ext, np.array(ctx.img_colorized))
            jpg_as_text = base64.b64encode(buffer)
            img = "data:image/" + overlay_ext + ";base64," + jpg_as_text.decode("utf-8")
        else:
            img = None
        return web.json_response({'details': results, 'img': img})

    class PostSchema(Schema):
        target_lang = fields.Str(required=False, validate=lambda a: a.upper() in VALID_LANGUAGES)
        detector = fields.Str(required=False, validate=lambda a: a.lower() in DETECTORS)
        ocr = fields.Str(required=False, validate=lambda a: a.lower() in OCRS)
        inpainter = fields.Str(required=False, validate=lambda a: a.lower() in INPAINTERS)
        upscaler = fields.Str(required=False, validate=lambda a: a.lower() in UPSCALERS)
        translator = fields.Str(required=False, validate=lambda a: a.lower() in TRANSLATORS)
        direction = fields.Str(required=False, validate=lambda a: a.lower() in {'auto', 'h', 'v'})
        skip_language = fields.Str(required=False)
        upscale_ratio = fields.Integer(required=False)
        translator_chain = fields.Str(required=False)
        selective_translation = fields.Str(required=False)
        attempts = fields.Integer(required=False)
        detection_size = fields.Integer(required=False)
        text_threshold = fields.Float(required=False)
        box_threshold = fields.Float(required=False)
        unclip_ratio = fields.Float(required=False)
        inpainting_size = fields.Integer(required=False)
        det_rotate = fields.Bool(required=False)
        det_auto_rotate = fields.Bool(required=False)
        det_invert = fields.Bool(required=False)
        det_gamma_correct = fields.Bool(required=False)
        min_text_length = fields.Integer(required=False)
        colorization_size = fields.Integer(required=False)
        denoise_sigma = fields.Integer(required=False)
        mask_dilation_offset = fields.Integer(required=False)
        ignore_bubble = fields.Integer(required=False)
        gpt_config = fields.String(required=False)
        filter_text = fields.String(required=False)

        # api specific
        overlay_ext = fields.Str(required=False)
        base64Images = fields.Raw(required=False)
        image = fields.Raw(required=False)
        url = fields.Raw(required=False)

        # no functionality except preventing errors when given
        fingerprint = fields.Raw(required=False)
        clientUuid = fields.Raw(required=False)