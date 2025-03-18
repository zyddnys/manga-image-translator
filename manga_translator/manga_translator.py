import asyncio
import cv2
import json
import langcodes
import langdetect
import os
import re
import time
import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional, Any

from .config import Config, Colorizer, Detector, Translator, Renderer, Inpainter
from .utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    load_image,
    dump_image,
    visualize_textblocks,
    is_valuable_text,
    sort_regions,
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection, unload as unload_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, unload as unload_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr, unload as unload_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting, unload as unload_inpainting
from .translators import (
    LANGDETECT_MAP,
    dispatch as dispatch_translation,
    prepare as prepare_translation,
    unload as unload_translation,
)
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')


def set_main_logger(l):
    global logger
    logger = l


class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass


def load_dictionary(file_path):
    dictionary = []
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                # Ignore empty lines and lines starting with '#' or '//'
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                # Remove comment parts
                line = line.split('#')[0].strip()
                line = line.split('//')[0].strip()
                parts = line.split()
                if len(parts) == 1:
                    # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, ''))
                elif len(parts) == 2:
                    # If both left and right parts are present, perform the replacement
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, parts[1]))
                else:
                    logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value in dictionary:
        text = pattern.sub(value, text)
    return text


class MangaTranslator:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str

    def __init__(self, params: dict = None):
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_path = None
        self.use_mtpe = False
        self.kernel_size = None
        self.device = None
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0

        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._detector_cleanup_task = None
        self.prep_manual = params.get('prep_manual', None)
        
    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.use_mtpe = params.get('use_mtpe', False)
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)

        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n'
                'Is the correct pytorch version installed? (See https://pytorch.org/)')
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        #todo: fix why is kernel size loaded in the constructor
        self.kernel_size=int(params.get('kernel_size'))
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config) -> Context:
        """
        Translates a PIL image from a manga. Returns dict with result and intermediates of translation.
        Default params are taken from args.py.

        ```py
        translation_dict = await translator.translate(image)
        result = translation_dict.result
        ```
        """
        # TODO: Take list of images to speed up batch processing

        ctx = Context()

        ctx.input = image
        ctx.result = None

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # translate
        return await self._translate(config, ctx)

    async def _translate(self, config: Config, ctx: Context) -> Context:
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            ctx.img_colorized = await self._run_colorizer(config, ctx)
        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            ctx.upscaled = await self._run_upscaling(config, ctx)
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        ctx.textlines = await self._run_ocr(config, ctx)

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # Apply pre-dictionary after OCR
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []  
        for textline in ctx.textlines:  
            original = textline.text  
            textline.text = apply_dictionary(textline.text, pre_dict)
            if original != textline.text:  
                pre_replacements.append(f"{original} => {textline.text}")  

        if pre_replacements:  
            logger.info("Pre-translation replacements:")  
            for replacement in pre_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No pre-translation replacements made.")
        
        # -- Textline merge
        await self._report_progress('textline_merge')
        ctx.text_regions = await self._run_textline_merge(config, ctx)

        if self.verbose:
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        # -- Translation
        await self._report_progress('translating')
        ctx.text_regions = await self._run_text_translation(config, ctx)
        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        # (Delayed to take advantage of the region filtering done after ocr and translation)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            ctx.mask = await self._run_mask_refinement(config, ctx)

        if self.verbose:
            inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), ctx.mask)

        # -- Inpainting
        await self._report_progress('inpainting')
        ctx.img_inpainted = await self._run_inpainting(config, ctx)
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            cv2.imwrite(self._result_path('inpainted.png'), cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
        # -- Rendering
        await self._report_progress('rendering')
        ctx.img_rendered = await self._run_text_rendering(config, ctx)

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)
    
    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        return ctx

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("colorizer", config.colorizer.colorizer)] = current_time
        #todo: im pretty sure the ctx is never used. does it need to be passed in?
        return await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            **ctx
        )

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("upscaling", config.upscale.upscaler)] = current_time
        return (await dispatch_upscaling(config.upscale.upscaler, [ctx.img_colorized], config.upscale.upscale_ratio, self.device))[0]

    async def _run_detection(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("detection", config.detector.detector)] = current_time
        return await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                        config.detector.box_threshold,
                                        config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                        config.detector.det_auto_rotate,
                                        self.device, self.verbose)

    async def _unload_model(self, tool: str, model: str):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorization':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'inpainting':
                await unload_inpainting(model)
            case 'ocr':
                await unload_ocr(model)
            case 'upscaling':
                await unload_upscaling(model)
            case 'translation':
                await unload_translation(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # empty CUDA cache

    # Background models cleanup job.
    async def _detector_cleanup_job(self):
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                if now - last_used > self.models_ttl:
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    async def _run_ocr(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("ocr", config.ocr.ocr)] = current_time
        textlines = await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device, self.verbose)

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if config.render.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = config.render.font_color_fg
                if config.render.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = config.render.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("textline_merge", "textline_merge")] = current_time
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     verbose=self.verbose)
        # Filter out languages to skip  
        if config.translator.skip_lang is not None:  
            skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]  
            filtered_textlines = []  
            for txtln in ctx.textlines:  
                try:  
                    detected_lang = langdetect.detect(txtln.text)  
                    source_language = LANGDETECT_MAP.get(detected_lang.lower(), 'UNKNOWN').upper()  
                except Exception:  
                    source_language = 'UNKNOWN'  
    
                # Print detected source_language and whether it's in skip_langs  
                # logger.info(f'Detected source language: {source_language}, in skip_langs: {source_language in skip_langs}, text: "{txtln.text}"')  
    
                if source_language in skip_langs:  
                    logger.info(f'Filtered out: {txtln.text}')  
                    logger.info(f'Reason: Detected language {source_language} is in skip_langs')  
                    continue  # Skip this region  
                filtered_textlines.append(txtln)  
            ctx.textlines = filtered_textlines  
    
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],  
                                                     verbose=self.verbose)  

        new_text_regions = []
        for region in text_regions:
            # Remove leading spaces after pre-translation dictionary replacement                
            original_text = region.text  
            stripped_text = original_text.strip()  
            
            # Record removed leading characters  
            removed_start_chars = original_text[:len(original_text) - len(stripped_text)]  
            if removed_start_chars:  
                logger.info(f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')  
            
            # Modified filtering condition: handle incomplete parentheses  
            # Combine left parentheses and left quotation marks into one list  
            left_symbols = ['(', '（', '[', '【', '{', '〔', '〈', '「',  
                            '“', '‘', '《', '『', '"', '〝', '﹁', '﹃',  
                            '⸂', '⸄', '⸉', '⸌', '⸜', '⸠', '‹', '«']  
            
            # Combine right parentheses and right quotation marks into one list
            right_symbols = [')', '）', ']', '】', '}', '〕', '〉', '」',  
                             '”', '’', '》', '』', '"', '〞', '﹂', '﹄',  
                             '⸃', '⸅', '⸊', '⸍', '⸝', '⸡', '›', '»']  
            # Combine all symbols  
            all_symbols = left_symbols + right_symbols  
            
            # Count the number of left and right symbols  
            left_count = sum(stripped_text.count(s) for s in left_symbols)  
            right_count = sum(stripped_text.count(s) for s in right_symbols)  
            
            # Check if the number of left and right symbols match  
            if left_count != right_count:  
                # Symbols don't match, remove all symbols  
                for s in all_symbols:  
                    stripped_text = stripped_text.replace(s, '')  
                logger.info(f'Removed unpaired symbols from "{stripped_text}"')  
              
            region.text = stripped_text.strip()     
            
            if len(region.text) >= config.ocr.min_text_length \
                    and not is_valuable_text(region.text) \
                    or (not config.translator.no_text_lang_skip and langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0):
                if region.text.strip():
                    logger.info(f'Filtered out: {region.text}')
                    if len(region.text) < config.ocr.min_text_length:
                        logger.info('Reason: Text length is less than the minimum required length.')
                    elif not is_valuable_text(region.text):
                        logger.info('Reason: Text is not considered valuable.')
                    elif langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0:
                        logger.info('Reason: Text language matches the target language and no_text_lang_skip is False.')
            else:
                if config.render.font_color_fg or config.render.font_color_bg:
                    if config.render.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        text_regions = new_text_regions


        # Sort ctd (comic text detector) regions left to right. Otherwise right to left.
        # Sorting will improve text translation quality.
        text_regions = sort_regions(text_regions, right_to_left=True if config.detector.detector != Detector.ctd else False)
        return text_regions

    async def _run_text_translation(self, config: Config, ctx: Context):
        # 如果设置了prep_manual则将translator设置为none，防止token浪费
        # Set translator to none to provent token waste if prep_manual is True  
        if self.prep_manual:  
            config.translator.translator = Translator.none          
    
        current_time = time.time()
        self._model_usage_timestamps[("translation", config.translator.translator)] = current_time

        # 为none翻译器添加特殊处理  
        # Add special handling for none translator  
        if config.translator.translator == Translator.none:  
            # 使用none翻译器时，为所有文本区域设置必要的属性  
            # When using none translator, set necessary properties for all text regions  
            for region in ctx.text_regions:  
                region.translation = ""  # 空翻译将创建空白区域 / Empty translation will create blank areas  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction   
            
            # 如果有prep_manual标志，则保留所有文本区域不进行过滤  
            # If prep_manual flag is present, keep all text regions without filtering  
            if self.prep_manual:  
                return ctx.text_regions  
            # 如果没有prep_manual标志，继续执行后续代码进行过滤  
            # If no prep_manual flag, continue to filtering logic below  

        # 以下翻译处理仅在非none翻译器或有none翻译器但没有prep_manual时执行  
        # Translation processing below only happens for non-none translator or none translator without prep_manual  
        if self.load_text:  
            input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
            with open(self._result_path(f"{input_filename}_translations.txt"), "r") as f:  
                    translated_sentences = json.load(f)  
        else:  
            # 如果是none翻译器，不需要调用翻译服务，文本已经设置为空  
            # If using none translator, no need to call translation service, text is already set to empty  
            if config.translator.translator != Translator.none:  
                translated_sentences = \
                    await dispatch_translation(config.translator.translator_gen,  
                                              [region.text for region in ctx.text_regions],  
                                              config.translator,  
                                              self.use_mtpe,  
                                              ctx, 'cpu' if self._gpu_limited_memory else self.device)  
            else:  
                # 对于none翻译器，创建一个空翻译列表  
                # For none translator, create an empty translation list  
                translated_sentences = ["" for _ in ctx.text_regions]  

            # Save translation if args.save_text is set and quit  
            if self.save_text:  
                input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
                with open(self._result_path(f"{input_filename}_translations.txt"), "w") as f:  
                    json.dump(translated_sentences, f, indent=4, ensure_ascii=False)  
                print("Don't continue if --save-text is used")  
                exit(-1)  

        # 如果不是none翻译器或者是none翻译器但没有prep_manual  
        # If not none translator or none translator without prep_manual  
        if config.translator.translator != Translator.none or not self.prep_manual:  
            for region, translation in zip(ctx.text_regions, translated_sentences):  
                if config.render.uppercase:  
                    translation = translation.upper()  
                elif config.render.lowercase:  
                    translation = translation.lower()  # 修正：应该是lower而不是upper  
                region.translation = translation  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction  

        # Punctuation correction logic. for translators often incorrectly change quotation marks from the source language to those commonly used in the target language.
        check_items = [
            ["(", "（", "「"],
            ["（", "(", "「"],
            [")", "）", "」"],
            ["）", ")", "」"],
            ["「", "“", "‘", "『"],
            ["」", "”", "’", "』"],
            ["『", "“", "‘", "「"],
            ["』", "”", "’", "」"],
        ]
        
        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "’"],
        ]
        
        for region in ctx.text_regions:
            if region.text and region.translation:
                # Detect 「」 or 『』 in the source text
                if '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                else:
                    quote_type = None
        
                # If the source text has 「」 or 『』, and the translation has "", replace them
                if quote_type and '"' in region.translation:
                    # Replace "" with 「」 or 『』
                    if quote_type == '「」':
                        region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                    elif quote_type == '『』':
                        region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
        
                # Correct ellipsis
                region.translation = re.sub(r'\.{3}', '…', region.translation)
        
                # Check and replace other symbols
                for v in check_items:
                    num_s = region.text.count(v[0])
                    num_t = sum(region.translation.count(t) for t in v[1:])
                    if num_s == num_t:
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # Apply post dictionary after translating
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")  

        # Filter out regions by their translations  
        new_text_regions = []  

        # List of languages with specific language detection  
        special_langs = ['CHS', 'CHT', 'KOR', 'IND', 'UKR', 'RUS', 'THA', 'ARA']  

        # Process special language scenarios  
        if config.translator.target_lang in special_langs:
            # Categorize regions  
            same_target_regions = []    # Target language regions with identical translation  
            diff_target_regions = []    # Target language regions with different translation  
            same_non_target_regions = []  # Non-target language regions with identical translation  
            diff_non_target_regions = []  # Non-target language regions with different translation  
            has_target_lang_in_translation_regions = []

            for region in ctx.text_regions:  
                        
                text_equal = region.text.lower().strip() == region.translation.lower().strip()  
                has_target_lang = False  
                has_target_lang_in_translation = False

                # Target language detection  
                if config.translator.target_lang in ['CHS', 'CHT']:  # Chinese
                    has_target_lang = bool(re.search('[\u4e00-\u9fff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u4e00-\u9fff]', region.translation))
                elif config.translator.target_lang == 'JPN':  # Japanese
                    has_target_lang = bool(re.search('[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', region.translation))
                elif config.translator.target_lang == 'KOR':  # Korean
                    has_target_lang = bool(re.search('[\uac00-\ud7af\u1100-\u11ff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\uac00-\ud7af\u1100-\u11ff]', region.translation))
                elif config.translator.target_lang == 'ARA':  # Arabic
                    has_target_lang = bool(re.search('[\u0600-\u06ff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u0600-\u06ff]', region.translation))
                elif config.translator.target_lang == 'THA':  # Thai
                    has_target_lang = bool(re.search('[\u0e00-\u0e7f]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u0e00-\u0e7f]', region.translation))
                elif config.translator.target_lang == 'RUS':  # Russian
                    has_target_lang = bool(re.search('[\u0400-\u04ff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u0400-\u04ff]', region.translation))
                elif config.translator.target_lang == 'UKR':  # Ukrainian
                    has_target_lang = bool(re.search('[\u0400-\u04ff]', region.text))
                    has_target_lang_in_translation = bool(re.search('[\u0400-\u04ff]', region.translation))
                elif config.translator.target_lang == 'IND':  # Indonesian
                    has_target_lang = bool(re.search('[A-Za-z]', region.text))
                    has_target_lang_in_translation = bool(re.search('[A-Za-z]', region.translation))

                # Skip numeric translations and filtered text  
                if region.translation.isnumeric():  
                    logger.info(f'Filtered out: {region.translation}')  
                    logger.info('Reason: Numeric translation')  
                    continue  
                
                if config.filter_text and re.search(config.re_filter_text, region.translation):
                    logger.info(f'Filtered out: {region.translation}')  
                    logger.info(f'Reason: Matched filter text: {config.filter_text}')
                    continue  
                
                if has_target_lang:  
                    if text_equal:
                        same_target_regions.append(region)  
                    else:  
                        diff_target_regions.append(region)  
                else:  
                    if text_equal:
                        same_non_target_regions.append(region)  
                    else:  
                        diff_non_target_regions.append(region)  

                if has_target_lang_in_translation:
                        has_target_lang_in_translation_regions.append(region)

            # If any different translations exist, retain all target language regions  
            if diff_target_regions or diff_non_target_regions:  
                new_text_regions.extend(same_target_regions)  
                new_text_regions.extend(diff_target_regions)  

            # Keep all non_target_lang regions with different translations (if translation contains target language characters)
            for region in diff_non_target_regions:  
                if region in has_target_lang_in_translation_regions:  
                    new_text_regions.append(region)  
                else:  
                    if config.translator.translator == Translator.none and not region.translation.strip():  
                        logger.info(f'Filtered out: {region.translation}')  
                        logger.info('Reason: Translation contain blank areas')  
                    else:  
                        logger.info(f'Filtered out: {region.translation}')  
                        logger.info('Reason: Translation does not contain target language characters')  

            # No different translations exist, clear all content.
            if not (diff_target_regions or diff_non_target_regions):
                for region in same_target_regions:
                    logger.info(f'Filtered out: {region.translation}')
                    logger.info('Reason: Translation identical to original -the whole page-')

            # Clear non_target_lang_regions with identical translations.
            for region in same_non_target_regions:
                logger.info(f'Filtered out: {region.translation}')
                logger.info('Reason: Translation identical to original -one textine-')


        else:  
            # Process non-special language scenarios using original logic  
            for region in ctx.text_regions:  
                    
                should_filter = False  
                filter_reason = ""  

                # 优先检查空白翻译 / Prioritize checking for blank translations  
                if not region.translation.strip():  
                    should_filter = True  
                    filter_reason = "Translation contain blank areas" 
                
                elif config.translator.translator != Translator.none:
                    if region.translation.isnumeric():  
                        should_filter = True  
                        filter_reason = "Numeric translation"  
                    elif config.filter_text and re.search(config.re_filter_text, region.translation):
                        should_filter = True  
                        filter_reason = f"Matched filter text: {config.filter_text}"
                    elif not config.translator.translator == Translator.original:
                        text_equal = region.text.lower().strip() == region.translation.lower().strip()  
                        if text_equal:  
                            should_filter = True  
                            filter_reason = "Translation identical to original"  
                
                if should_filter:  
                    if region.translation.strip():  
                        logger.info(f'Filtered out: {region.translation}')  
                        logger.info(f'Reason: {filter_reason}')  
                else:  
                    new_text_regions.append(region)  

        return new_text_regions 
               

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              config.mask_dilation_offset, config.ocr.ignore_bubble, self.verbose,self.kernel_size)

    async def _run_inpainting(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("inpainting", config.inpainter.inpainter)] = current_time
        return await dispatch_inpainting(config.inpainter.inpainter, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size, self.device,
                                         self.verbose)

    async def _run_text_rendering(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("rendering", config.render.renderer)] = current_time
        if config.render.renderer == Renderer.none:
            output = ctx.img_inpainted
        # manga2eng currently only supports horizontal left to right rendering
        elif config.render.renderer == Renderer.manga2Eng and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(
                ctx.text_regions[0].target_lang) == 'h':
            output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
        else:
            output = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, self.font_path, config.render.font_size,
                                              config.render.font_size_offset,
                                              config.render.font_size_minimum, not config.render.no_hyphenation, ctx.render_mask, config.render.line_spacing)
        return output

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
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
