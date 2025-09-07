import asyncio
import cv2
import json
import langcodes
import os
import regex as re
import time
import torch
import logging
import sys
import traceback
import numpy as np
from PIL import Image
from typing import Optional, Any, List
import py3langid as langid

from .config import Config, Colorizer, Detector, Translator, Renderer, Inpainter, PanelDetector
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
from .panel_detection import prepare as prepare_panel_detection, unload as unload_panel_detection
from .translators import (
    dispatch as dispatch_translation,
    prepare as prepare_translation,
    unload as unload_translation,
)
from .translators.common import ISO_639_1_TO_VALID_LANGUAGES
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_eng_render_pillow

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

# 全局console实例，用于日志重定向
_global_console = None
_log_console = None

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
                    dictionary.append((pattern, '', line_number))
                elif len(parts) == 2:
                    # If both left and right parts are present, perform the replacement
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, parts[1], line_number))
                else:
                    logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        original_text = text
        text = pattern.sub(value, text)
        if text != original_text:
            logger.info(f'Line {line_number}: Replaced "{original_text}" with "{text}" using pattern "{pattern.pattern}" and value "{value}"')
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
    batch_size: int

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
        self.batch_size = 1  # 默认不批量处理

        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        
        self._batch_contexts = []  # 存储批量处理的上下文
        self._batch_configs = []   # 存储批量处理的配置
        self.disable_memory_optimization = params.get('disable_memory_optimization', False)
        # batch_concurrent 会在 parse_init_params 中验证并设置
        self.batch_concurrent = params.get('batch_concurrent', False)
        
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
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []
        self._original_page_texts = []  # 存储原文页面数据，用于并发模式下的上下文

        # 调试图片管理相关属性
        self._current_image_context = None  # 存储当前处理图片的上下文信息
        self._saved_image_contexts = {}     # 存储批量处理中每个图片的上下文信息
        
        # 设置日志文件
        self._setup_log_file()

    def _setup_log_file(self):
        """设置日志文件，在result文件夹下创建带时间戳的log文件"""
        try:
            # 创建result目录
            result_dir = os.path.join(BASE_PATH, 'result')
            os.makedirs(result_dir, exist_ok=True)
            
            # 生成带时间戳的日志文件名
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            log_filename = f"log_{timestamp}.txt"
            log_path = os.path.join(result_dir, log_filename)
            
            # 配置文件日志处理器
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            # 使用自定义格式器，保持与控制台输出一致
            from .utils.log import Formatter
            formatter = Formatter()
            file_handler.setFormatter(formatter)
            
            # 添加到manga-translator根logger以捕获所有输出
            mt_logger = logging.getLogger('manga-translator')
            mt_logger.addHandler(file_handler)
            if not mt_logger.level or mt_logger.level > logging.DEBUG:
                mt_logger.setLevel(logging.DEBUG)
            
            # 保存日志文件路径供后续使用
            self._log_file_path = log_path
            
            # 简单的print重定向
            import builtins
            original_print = builtins.print
            
            def log_print(*args, **kwargs):
                # 正常打印到控制台
                original_print(*args, **kwargs)
                # 同时写入日志文件
                try:
                    import io
                    buffer = io.StringIO()
                    original_print(*args, file=buffer, **kwargs)
                    output = buffer.getvalue()
                    if output.strip():
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(output)
                except Exception:
                    pass
            
            builtins.print = log_print
            
            # Rich Console输出重定向
            try:
                from rich.console import Console
                import sys
                
                # 创建一个自定义的文件对象，同时写入控制台和日志文件
                class TeeFile:
                    def __init__(self, log_file_path, original_file):
                        self.log_file_path = log_file_path
                        self.original_file = original_file
                    
                    def write(self, text):
                        # 写入原始输出
                        self.original_file.write(text)
                        # 写入日志文件
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        self.original_file.flush()
                    
                    def __getattr__(self, name):
                        return getattr(self.original_file, name)
                
                # 创建一个仅用于日志记录的Console（无颜色、无样式）
                class LogOnlyFile:
                    def __init__(self, log_file_path):
                        self.log_file_path = log_file_path
                    
                    def write(self, text):
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        pass
                    
                    def isatty(self):
                        return False
                
                # 为日志创建纯文本console
                log_file_only = LogOnlyFile(log_path)
                log_console = Console(file=log_file_only, force_terminal=False, no_color=True, width=80)
                
                # 创建带颜色的控制台console
                display_console = Console(force_terminal=True)
                
                # 全局设置console实例，供translator使用
                global _global_console, _log_console
                _global_console = display_console  # 控制台显示用
                _log_console = log_console         # 日志记录用
                
            except Exception as e:
                logger.debug(f"Failed to setup rich console logging: {e}")
            
            logger.info(f"Log file created: {log_path}")
        except Exception as e:
            print(f"Failed to setup log file: {e}")

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.use_mtpe = params.get('use_mtpe', False)
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)
        self.batch_size = params.get('batch_size', 1)  # 添加批量大小参数
        
        # 验证batch_concurrent参数
        if self.batch_concurrent and self.batch_size < 2:
            logger.warning('--batch-concurrent requires --batch-size to be at least 2. When batch_size is 1, concurrent mode has no effect.')
            logger.info('Suggestion: Use --batch-size 2 (or higher) with --batch-concurrent, or remove --batch-concurrent flag.')
            # 自动禁用并发模式
            self.batch_concurrent = False
            
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
        
        # batch_concurrent 已在初始化时设置并验证
        

        
    def _set_image_context(self, config: Config, image=None):
        """设置当前处理图片的上下文信息，用于生成调试图片子文件夹"""
        from .utils.generic import get_image_md5

        # 使用毫秒级时间戳确保唯一性
        timestamp = str(int(time.time() * 1000))
        detection_size = str(getattr(config.detector, 'detection_size', 1024))
        target_lang = getattr(config.translator, 'target_lang', 'unknown')
        translator = getattr(config.translator, 'translator', 'unknown')

        # 计算图片MD5哈希值
        if image is not None:
            file_md5 = get_image_md5(image)
        else:
            file_md5 = "unknown"

        # 生成子文件夹名：{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}
        subfolder_name = f"{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}"

        self._current_image_context = {
            'subfolder': subfolder_name,
            'file_md5': file_md5,
            'config': config
        }
        
    def _get_image_subfolder(self) -> str:
        """获取当前图片的调试子文件夹名"""
        if self._current_image_context:
            return self._current_image_context['subfolder']
        return ''
    
    def _save_current_image_context(self, image_md5: str):
        """保存当前图片上下文，用于批量处理中保持一致性"""
        if self._current_image_context:
            self._saved_image_contexts[image_md5] = self._current_image_context.copy()

    def _restore_image_context(self, image_md5: str):
        """恢复保存的图片上下文"""
        if image_md5 in self._saved_image_contexts:
            self._current_image_context = self._saved_image_contexts[image_md5].copy()
            return True
        return False

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config, image_name: str = None, skip_context_save: bool = False) -> Context:
        """
        Translates a single image.

        :param image: Input image.
        :param config: Translation config.
        :param image_name: Deprecated parameter, kept for compatibility.
        :return: Translation context.
        """
        await self._report_progress('running_pre_translation_hooks')
        for hook in self._progress_hooks:
            try:
                hook('running_pre_translation_hooks', False)
            except Exception as e:
                logger.error(f"Error in progress hook: {e}")

        ctx = Context()
        ctx.input = image
        ctx.result = None
        ctx.verbose = self.verbose

        # 设置图片上下文以生成调试图片子文件夹
        self._set_image_context(config, image)
        
        # 保存debug文件夹信息到Context中（用于Web模式的缓存访问）
        # 在web模式下总是保存，不仅仅是verbose模式
        ctx.debug_folder = self._get_image_subfolder()
        
        # 保存原始输入图片用于调试
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png')
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {result_path}")
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            if config.panel_detector.panel_detector != 'none':
                await prepare_panel_detection(config.panel_detector.panel_detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # translate
        ctx = await self._translate(config, ctx)

        # 在翻译流程的最后保存翻译结果，确保保存的是最终结果（包括重试后的结果）
        # Save translation results at the end of translation process to ensure final results are saved
        if not skip_context_save and ctx.text_regions:
            # 汇总本页翻译，供下一页做上文
            page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                 for r in ctx.text_regions}
            self.all_page_translations.append(page_translations)

            # 同时保存原文用于并发模式的上下文
            page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                  for i, r in enumerate(ctx.text_regions)}
            self._original_page_texts.append(page_original_texts)

        return ctx

    async def _translate(self, config: Config, ctx: Context) -> Context:
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input  # Fallback to input image if colorization fails

        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized # Fallback to colorized (or input) image if upscaling fails
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
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
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] # Fallback to empty textlines if OCR fails

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if textline merge fails

        if self.verbose and ctx.text_regions:
            bbox_start_time = time.time()
            show_panels = config.panel_detector.panel_detector != 'none'  # 当启用分镜检测时显示panel | Show panel when panel detection is enabled
            bboxes = await visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions,
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl, device=self.device, panel_detector=config.panel_detector.panel_detector, panel_config=config.panel_detector, ctx=ctx)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        # Apply pre-dictionary after textline merge
        await self._apply_pre_translation_replacements(ctx)
            
        # -- Translation
        await self._report_progress('translating')
        try:
            ctx.text_regions = await self._run_text_translation(config, ctx)
        except Exception as e:  
            logger.error(f"Error during translating:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if translation fails

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
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0] # Fallback to raw mask or empty mask

        if self.verbose and ctx.mask is not None:
            inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), ctx.mask)

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
        except Exception as e:  
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"Failed to save debug image: {inpainted_path}")
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted # Fallback to inpainted (or original RGB) image if rendering fails

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)

    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        # 在verbose模式下保存final.png到调试文件夹
        if ctx.result and self.verbose:
            try:
                final_img = np.array(ctx.result)
                if len(final_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                final_path = self._result_path('final.png')
                success = cv2.imwrite(final_path, final_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {final_path}")
            except Exception as e:
                logger.error(f"Error saving final.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # Web流式模式优化：保存final.png并使用占位符
        if ctx.result and not self.result_sub_folder and hasattr(self, '_is_streaming_mode') and self._is_streaming_mode:
            # 保存final.png文件
            final_img = np.array(ctx.result)
            if len(final_img.shape) == 3:  # 彩色图片，转换BGR顺序
                final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._result_path('final.png'), final_img)

            # 通知前端文件已就绪
            if hasattr(self, '_progress_hooks') and self._current_image_context:
                folder_name = self._current_image_context['subfolder']
                await self._report_progress(f'final_ready:{folder_name}')

            # 创建占位符结果并立即返回
            from PIL import Image
            placeholder = Image.new('RGB', (1, 1), color='white')
            ctx.result = placeholder
            ctx.use_placeholder = True
            return ctx

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
        result = await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                        config.detector.box_threshold,
                                        config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                        config.detector.det_auto_rotate,
                                        self.device, self.verbose)
        return result



    async def _unload_model(self, tool: str, model: str):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorization':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'panel_detection':
                await unload_panel_detection(model)
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
        
        # 为OCR创建子文件夹（只在verbose模式下）
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, 'ocrs')
                else:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', image_subfolder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
            else:
                ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
        else:
            # 非verbose模式下使用临时目录或不创建OCR结果目录
            ocr_result_dir = None
        
        # 临时设置环境变量供OCR模块使用
        old_ocr_dir = os.environ.get('MANGA_OCR_RESULT_DIR', None)
        if ocr_result_dir:
            os.environ['MANGA_OCR_RESULT_DIR'] = ocr_result_dir
        
        try:
            textlines = await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device, self.verbose)
        finally:
            # 恢复环境变量
            if old_ocr_dir is not None:
                os.environ['MANGA_OCR_RESULT_DIR'] = old_ocr_dir
            elif 'MANGA_OCR_RESULT_DIR' in os.environ:
                del os.environ['MANGA_OCR_RESULT_DIR']

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
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                region.text_raw = region.text      # <- Save the initial OCR results to expand the render detection box. Also, prevent affecting the forbidden translation function.       
        # Filter out languages to skip  
        if config.translator.skip_lang is not None:  
            skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]  
            filtered_textlines = []  
            for txtln in ctx.textlines:  
                try:  
                    detected_lang, confidence = langid.classify(txtln.text)
                    source_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
                    if source_language != 'UNKNOWN':
                        source_language = source_language.upper()
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
            bracket_pairs = {  
                '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',  
                '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '"': '"', '〝': '〞', '﹁': '﹂', '﹃': '﹄',  
                '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'  
            }   
            left_symbols = set(bracket_pairs.keys())  
            right_symbols = set(bracket_pairs.values())  
            
            has_brackets = any(s in stripped_text for s in left_symbols) or any(s in stripped_text for s in right_symbols)  
            
            if has_brackets:  
                result_chars = []  
                stack = []  
                to_skip = []    
                
                # 第一次遍历：标记匹配的括号  
                # First traversal: mark matching brackets
                for i, char in enumerate(stripped_text):  
                    if char in left_symbols:  
                        stack.append((i, char))  
                    elif char in right_symbols:  
                        if stack:  
                            # 有对应的左括号，出栈  
                            # There is a corresponding left bracket, pop the stack
                            stack.pop()  
                        else:  
                            # 没有对应的左括号，标记为删除  
                            # No corresponding left parenthesis, marked for deletion
                            to_skip.append(i)  
                
                # 标记未匹配的左括号为删除
                # Mark unmatched left brackets as delete  
                for pos, _ in stack:  
                    to_skip.append(pos)  
                
                has_removed_symbols = len(to_skip) > 0  
                
                # 第二次遍历：处理匹配但不对应的括号
                # Second pass: Process matching but mismatched brackets
                stack = []  
                for i, char in enumerate(stripped_text):  
                    if i in to_skip:  
                        # 跳过孤立的括号
                        # Skip isolated parentheses
                        continue  
                        
                    if char in left_symbols:  
                        stack.append(char)  
                        result_chars.append(char)  
                    elif char in right_symbols:  
                        if stack:  
                            left_bracket = stack.pop()  
                            expected_right = bracket_pairs.get(left_bracket)  
                            
                            if char != expected_right:  
                                # 替换不匹配的右括号为对应左括号的正确右括号
                                # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                                result_chars.append(expected_right)  
                                logger.info(f'Fixed mismatched bracket: replaced "{char}" with "{expected_right}"')  
                            else:  
                                result_chars.append(char)  
                    else:  
                        result_chars.append(char)  
                
                new_stripped_text = ''.join(result_chars)  
                
                if has_removed_symbols:  
                    logger.info(f'Removed unpaired bracket from "{stripped_text}"')  
                
                if new_stripped_text != stripped_text and not has_removed_symbols:  
                    logger.info(f'Fixed brackets: "{stripped_text}" → "{new_stripped_text}"')  
                
                stripped_text = new_stripped_text  
              
            region.text = stripped_text.strip()     
            
            if len(region.text) < config.ocr.min_text_length \
                    or not is_valuable_text(region.text) \
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

        # Record panel detection usage for TTL management (only if not disabled)
        if config.panel_detector.panel_detector != 'none':
            current_time = time.time()
            self._model_usage_timestamps[("panel_detection", config.panel_detector.panel_detector)] = current_time

        text_regions = await sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb,
            device=self.device,
            panel_detector=config.panel_detector.panel_detector,
            panel_config=config.panel_detector,
            ctx=ctx
        )
        
        return text_regions

    async def _run_text_translation(self, config: Config, ctx: Context):
        # 检查text_regions是否为None或空
        if not ctx.text_regions:
            return []
            
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
            return ctx.text_regions  

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
                # 统一使用dispatch_translation调用所有翻译器
                # Use unified dispatch_translation for all translators
                texts = [region.text for region in ctx.text_regions]
                # 将上下文参数放入 ctx
                ctx.context_size = self.context_size
                ctx.all_page_translations = self.all_page_translations
                translated_sentences = await dispatch_translation(
                    config.translator.translator_gen,
                    texts,
                    config.translator,
                    self.use_mtpe,
                    ctx,
                    device=self.device
                )
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


        # 字典替换 | Dictionary replacement
        await self._apply_post_translation_dictionary(ctx)

        return ctx.text_regions

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
        elif (config.render.renderer == Renderer.manga2Eng or config.render.renderer == Renderer.manga2EngPillow) and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang) == 'h':
            if config.render.renderer == Renderer.manga2EngPillow:
                output = await dispatch_eng_render_pillow(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
            else:
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
        # 只有在verbose模式下才使用图片级子文件夹
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, path)
                else:
                    result_path = os.path.join(BASE_PATH, 'result', image_subfolder, path)
                # 确保目录存在
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                return result_path
        
        # 在server/web模式下（result_sub_folder为空）且为非verbose模式时
        # 需要创建一个子文件夹来保存final.png
        if not self.result_sub_folder:
            if self._current_image_context:
                # 直接使用已生成的子文件夹名
                sub_folder = self._current_image_context['subfolder']
            else:
                # 没有上下文信息时使用默认值
                timestamp = str(int(time.time() * 1000))
                sub_folder = f"{timestamp}-unknown-1024-unknown-unknown"

            result_path = os.path.join(BASE_PATH, 'result', sub_folder, path)
        else:
            result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        return result_path

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

    async def translate_batch(self, images_with_configs: List[tuple], batch_size: int = None, image_names: List[str] = None) -> List[Context]:
        """
        批量翻译多张图片，在翻译阶段进行批量处理以提高效率
        Args:
            images_with_configs: List of (image, config) tuples
            batch_size: 批量大小，如果为None则使用实例的batch_size
            image_names: 已弃用的参数，保留用于兼容性
        Returns:
            List of Context objects with translation results
        """
        batch_size = batch_size or self.batch_size
        if batch_size <= 1:
            # 不使用批量处理时，回到原来的逐个处理方式
            logger.debug('Batch size <= 1, switching to individual processing mode')
            results = []
            for i, (image, config) in enumerate(images_with_configs):
                ctx = await self.translate(image, config)  # 单页翻译时正常保存上下文
                results.append(ctx)
            return results
        
        
        # 简化的内存检查
        memory_optimization_enabled = not self.disable_memory_optimization
        if not memory_optimization_enabled:
            logger.debug('Memory optimization disabled for batch translation')
        
        results = []
        
        # 处理所有图片到翻译之前的步骤
        pre_translation_contexts = []
        
        for i, (image, config) in enumerate(images_with_configs):
            logger.debug(f'Pre-processing image {i+1}/{len(images_with_configs)}')
            
            # 简化的内存检查
            if memory_optimization_enabled:
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        logger.warning(f'High memory usage during pre-processing: {memory_percent:.1f}%')
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except ImportError:
                    pass  # psutil 不可用时忽略
                except Exception as e:
                    logger.debug(f'Memory check failed: {e}')
                
            try:
                # 为批量处理中的每张图片设置上下文
                self._set_image_context(config, image)
                # 保存图片上下文，确保后处理阶段使用相同的文件夹
                if self._current_image_context:
                    image_md5 = self._current_image_context['file_md5']
                    self._save_current_image_context(image_md5)
                ctx = await self._translate_until_translation(image, config)
                # 保存图片上下文到Context对象中，用于后续批量处理
                if self._current_image_context:
                    ctx.image_context = self._current_image_context.copy()
                # 保存verbose标志到Context对象中
                ctx.verbose = self.verbose
                pre_translation_contexts.append((ctx, config))
                logger.debug(f'Image {i+1} pre-processing successful')
            except MemoryError as e:
                logger.error(f'Memory error in pre-processing image {i+1}: {e}')
                raise
            except Exception as e:
                logger.error(f'Image {i+1} pre-processing error: {e}')
                # 创建空context作为占位符
                ctx = Context()
                ctx.input = image
                ctx.text_regions = []  # 确保text_regions被初始化为空列表
                pre_translation_contexts.append((ctx, config))
        
        if not pre_translation_contexts:
            logger.warning('No images pre-processed successfully')
            return results
            
        logger.debug(f'Pre-processing completed: {len(pre_translation_contexts)} images')
            
        # 批量翻译处理
        logger.debug('Starting batch translation phase...')
        try:
            if self.batch_concurrent:
                logger.info(f'Using concurrent mode for batch translation')
                translated_contexts = await self._concurrent_translate_contexts(pre_translation_contexts)
            else:
                logger.debug(f'Using standard batch mode for translation')
                translated_contexts = await self._batch_translate_contexts(pre_translation_contexts, batch_size)
        except MemoryError as e:
            logger.error(f'Memory error in batch translation: {e}')
            raise
        
        # 完成翻译后的处理
        logger.debug('Starting post-processing phase...')
        for i, (ctx, config) in enumerate(translated_contexts):
            try:
                if ctx.text_regions:
                    # 恢复预处理阶段保存的图片上下文，确保使用相同的文件夹
                    # 通过图片计算MD5来恢复上下文
                    from .utils.generic import get_image_md5
                    image = ctx.input  # 从context中获取原始图片
                    image_md5 = get_image_md5(image)
                    if not self._restore_image_context(image_md5):
                        # 如果恢复失败，作为fallback重新设置（理论上不应该发生）
                        logger.warning(f"Failed to restore image context for MD5 {image_md5}, creating new context")
                        self._set_image_context(config, image)
                    ctx = await self._complete_translation_pipeline(ctx, config)
                results.append(ctx)
                logger.debug(f'Image {i+1} post-processing completed')
            except Exception as e:
                logger.error(f'Image {i+1} post-processing error: {e}')
                results.append(ctx)
        
        logger.info(f'Batch translation completed: processed {len(results)} images')

        # 批处理完成后，保存所有页面的最终翻译结果 | After batch processing, save final translation results for all pages
        for ctx in results:
            if ctx.text_regions:
                # 汇总本页翻译，供下一页做上文 | Summarize current page translations for next page context
                page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                     for r in ctx.text_regions}
                self.all_page_translations.append(page_translations)

                # 同时保存原文用于并发模式的上下文 | Also save original text for concurrent mode context
                page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                      for i, r in enumerate(ctx.text_regions)}
                self._original_page_texts.append(page_original_texts)

        # 清理批量处理的图片上下文缓存
        self._saved_image_contexts.clear()
        
        return results

    async def _translate_until_translation(self, image: Image.Image, config: Config) -> Context:
        """
        执行翻译之前的所有步骤（彩色化、上采样、检测、OCR、文本行合并）
        """
        ctx = Context()
        ctx.input = image
        ctx.result = None
        
        # 保存原始输入图片用于调试
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png')
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {result_path}")
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            if config.panel_detector.panel_detector != 'none':
                await prepare_panel_detection(config.panel_detector.panel_detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input
        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = []

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = []

        if self.verbose and ctx.text_regions:
            bbox_start_time = time.time()
            show_panels = config.panel_detector.panel_detector != 'none'  # 当启用分镜检测时显示panel | Show panel when panel detection is enabled
            bboxes = await visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions,
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl, device=self.device, panel_detector=config.panel_detector.panel_detector, panel_config=config.panel_detector, ctx=ctx)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        # Apply pre-dictionary after textline merge
        await self._apply_pre_translation_replacements(ctx)

        # 保存当前图片上下文到ctx中，用于并发翻译时的路径管理
        if self._current_image_context:
            ctx.image_context = self._current_image_context.copy()

        return ctx

    async def _batch_translate_contexts(self, contexts_with_configs: List[tuple], batch_size: int) -> List[tuple]:
        """
        批量处理翻译步骤，防止内存溢出
        """
        results = []
        total_contexts = len(contexts_with_configs)
        
        # 按批次处理，防止内存溢出
        for i in range(0, total_contexts, batch_size):
            batch = contexts_with_configs[i:i + batch_size]
            logger.info(f'Processing translation batch {i//batch_size + 1}/{(total_contexts + batch_size - 1)//batch_size}')
            
            # 收集当前批次的所有文本
            all_texts = []
            batch_text_mapping = []  # 记录每个文本属于哪个context和region
            
            for ctx_idx, (ctx, config) in enumerate(batch):
                if not ctx.text_regions:
                    continue
                    
                region_start_idx = len(all_texts)
                for region_idx, region in enumerate(ctx.text_regions):
                    all_texts.append(region.text)
                    batch_text_mapping.append((ctx_idx, region_idx))
                
            if not all_texts:
                # 当前批次没有需要翻译的文本
                results.extend(batch)
                continue
                
            # 批量翻译
            try:
                await self._report_progress('translating')
                # 使用第一个配置进行翻译（假设批次内配置相同）
                sample_config = batch[0][1] if batch else None
                if sample_config:
                    # 支持批量翻译 - 传递所有批次上下文
                    batch_contexts = [ctx for ctx, config in batch]
                    translated_texts = await self._batch_translate_texts(all_texts, sample_config, batch[0][0], batch_contexts)
                else:
                    translated_texts = all_texts  # 无法翻译时保持原文
                    
                # 将翻译结果分配回各个context
                text_idx = 0
                for ctx_idx, (ctx, config) in enumerate(batch):
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region_idx, region in enumerate(ctx.text_regions):
                        if text_idx < len(translated_texts):
                            region.translation = translated_texts[text_idx]
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                            text_idx += 1

                # 字典替换 | Dictionary replacement
                for ctx, config in batch:
                    if ctx.text_regions:
                        await self._apply_post_translation_dictionary(ctx)
                        
                results.extend(batch)
                
            except Exception as e:
                logger.error(f"Error in batch translation: {e}")
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                for ctx, config in batch:
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                results.extend(batch)
                
            # 强制垃圾回收以释放内存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results

    async def _concurrent_translate_contexts(self, contexts_with_configs: List[tuple]) -> List[tuple]:
        """
        并发处理翻译步骤，为每个图片单独发送翻译请求，避免合并大批次
        """

        # 在并发模式下，先保存所有页面的原文用于上下文
        batch_original_texts = []  # 存储当前批次的原文
        if self.context_size > 0:
            for i, (ctx, config) in enumerate(contexts_with_configs):
                if ctx.text_regions:
                    # 保存当前页面的原文
                    page_texts = {}
                    for j, region in enumerate(ctx.text_regions):
                        page_texts[j] = region.text
                    batch_original_texts.append(page_texts)

                    # 确保 _original_page_texts 有足够的长度
                    while len(self._original_page_texts) <= len(self.all_page_translations) + i:
                        self._original_page_texts.append({})

                    self._original_page_texts[len(self.all_page_translations) + i] = page_texts
                else:
                    batch_original_texts.append({})

        async def translate_single_context(ctx_config_pair_with_index):
            """翻译单个context的异步函数"""
            ctx, config, page_index, batch_index = ctx_config_pair_with_index
            try:
                if not ctx.text_regions:
                    return ctx, config

                # 收集该context的所有文本
                texts = [region.text for region in ctx.text_regions]

                if not texts:
                    return ctx, config

                logger.debug(f'Translating {len(texts)} regions for single image in concurrent mode (page {page_index}, batch {batch_index})')

                # 单独翻译这一张图片的文本，传递页面索引和批次索引用于正确的上下文
                translated_texts = await self._batch_translate_texts(
                    texts, config, ctx,
                    page_index=page_index,
                    batch_index=batch_index,
                    batch_original_texts=batch_original_texts
                )

                # 将翻译结果分配回各个region
                for i, region in enumerate(ctx.text_regions):
                    if i < len(translated_texts):
                        region.translation = translated_texts[i]
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                

                # 字典替换 | Dictionary replacement
                if ctx.text_regions:
                    await self._apply_post_translation_dictionary(ctx)
                
                return ctx, config
                
            except Exception as e:
                logger.error(f"Error in concurrent translation for single image: {e}")
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                return ctx, config
        
        # 创建并发任务，为每个任务添加页面索引和批次索引
        tasks = []
        for i, ctx_config_pair in enumerate(contexts_with_configs):
            # 计算当前页面在整个翻译序列中的索引
            page_index = len(self.all_page_translations) + i
            batch_index = i  # 在当前批次中的索引
            ctx_config_pair_with_index = (*ctx_config_pair, page_index, batch_index)
            task = asyncio.create_task(translate_single_context(ctx_config_pair_with_index))
            tasks.append(task)
        
        logger.info(f'Starting concurrent translation of {len(tasks)} images...')
        
        # 等待所有任务完成
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in concurrent translation gather: {e}")
            raise
        
        # 处理结果，检查是否有异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Image {i+1} concurrent translation failed: {result}")
                if not self.ignore_errors:
                    raise result
                # 创建失败的占位符
                ctx, config = contexts_with_configs[i]
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                final_results.append((ctx, config))
            else:
                final_results.append(result)
        
        logger.info(f'Concurrent translation completed: {len(final_results)} images processed')
        return final_results

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context, batch_contexts: List[Context] = None, page_index: int = None, batch_index: int = None, batch_original_texts: List[dict] = None) -> List[str]:
        """
        批量翻译文本列表，使用现有的翻译器接口

        Args:
            texts: 要翻译的文本列表
            config: 配置对象
            ctx: 上下文对象
            batch_contexts: 批处理上下文列表
            page_index: 当前页面索引，用于并发模式下的上下文计算
            batch_index: 当前页面在批次中的索引
            batch_original_texts: 当前批次的原文数据
        """
        if config.translator.translator == Translator.none:
            return ["" for _ in texts]

        # 设置并发模式信息到ctx中，供翻译器使用
        if self.batch_concurrent and self.batch_size > 1:
            ctx._batch_concurrent = True
            ctx._page_index = page_index
            ctx._batch_index = batch_index
            ctx._batch_original_texts = batch_original_texts
            ctx._original_page_texts = getattr(self, '_original_page_texts', [])


        # 为并发翻译设置特殊的上下文信息
        if config.translator.translator == Translator.chatgpt_2stage:
            # 为当前图片创建专用的result_path_callback，避免并发时路径错位
            current_image_context = getattr(ctx, 'image_context', None) or self._current_image_context

            def result_path_callback(path: str) -> str:
                """为特定图片创建结果路径，使用保存的图片上下文"""
                original_context = self._current_image_context
                self._current_image_context = current_image_context
                try:
                    return self._result_path(path)
                finally:
                    self._current_image_context = original_context

            ctx.result_path_callback = result_path_callback

            # Check if batch processing is enabled and batch_contexts are provided
            if batch_contexts and len(batch_contexts) > 1 and not self.batch_concurrent:
                # Enable batch processing for chatgpt_2stage
                ctx.batch_contexts = batch_contexts
                logger.info(f"Enabling batch processing for chatgpt_2stage with {len(batch_contexts)} images")

                # Set result_path_callback for each context in the batch
                for batch_ctx in batch_contexts:
                    if hasattr(batch_ctx, 'image_context'):
                        batch_image_context = batch_ctx.image_context
                    else:
                        batch_image_context = self._current_image_context

                    def create_result_path_callback(image_context):
                        def result_path_callback(path: str) -> str:
                            """为特定图片创建结果路径，使用保存的图片上下文"""
                            original_context = self._current_image_context
                            self._current_image_context = image_context
                            try:
                                return self._result_path(path)
                            finally:
                                self._current_image_context = original_context
                        return result_path_callback

                    batch_ctx.result_path_callback = create_result_path_callback(batch_image_context)

        # 统一使用dispatch_translation调用所有翻译器
        # 将上下文参数放入 ctx
        ctx.context_size = self.context_size
        ctx.all_page_translations = self.all_page_translations
        return await dispatch_translation(
            config.translator.translator_gen,
            texts,
            config.translator,
            self.use_mtpe,
            ctx,
            'cpu' if self._gpu_limited_memory else self.device
        )
            

    async def _complete_translation_pipeline(self, ctx: Context, config: Config) -> Context:
        """
        完成翻译后的处理步骤（掩码细化、修复、渲染）
        """
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
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0]

        if self.verbose and ctx.mask is not None:
            try:
                inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                              self.device, self.verbose)
                
                # 保存inpaint_input.png
                inpaint_input_path = self._result_path('inpaint_input.png')
                success1 = cv2.imwrite(inpaint_input_path, cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
                if not success1:
                    logger.warning(f"Failed to save debug image: {inpaint_input_path}")
                
                # 保存mask_final.png
                mask_final_path = self._result_path('mask_final.png')
                success2 = cv2.imwrite(mask_final_path, ctx.mask)
                if not success2:
                    logger.warning(f"Failed to save debug image: {mask_final_path}")
            except Exception as e:
                logger.error(f"Error saving debug images (inpaint_input.png, mask_final.png): {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)

        except Exception as e:  
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"Failed to save debug image: {inpainted_path}")
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
        
        # 保存debug文件夹信息到Context中（用于Web模式的缓存访问）
        if self.verbose:
            ctx.debug_folder = self._get_image_subfolder()

        return await self._revert_upscale(config, ctx)

    async def _apply_pre_translation_replacements(self, ctx: Context):
        """
        应用译前替换（字典替换） | Apply pre-translation replacements (dictionary replacement)
        """
        if not ctx.text_regions:
            return

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")

    async def _apply_post_translation_dictionary(self, ctx: Context):
        """
        应用译后字典替换 | Apply post-translation dictionary replacement
        """
        if not ctx.text_regions:
            return

        # 应用后字典 | Apply post-translation dictionary
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