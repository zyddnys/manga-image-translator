import json
import os
import gc
import copy
from typing import Union, List
import time  

from PIL import Image
import psutil

from manga_translator import MangaTranslator, Context, TranslationInterrupt, Config
from ..save import save_result
from ..translators import (
    LanguageUnsupportedException,
    dispatch as dispatch_translation,
)
from ..utils import natural_sort, replace_prefix, get_color_name, rgb2hex, get_logger

# 使用专用的local logger
logger = get_logger('local')

# 提示音开关
ENABLE_COMPLETION_SOUND = True

def play_completion_sound():
    """播放完成提示音"""
    try:
        import platform
        if platform.system() == 'Windows':
            import winsound
            # 使用默认系统提示音
            winsound.MessageBeep(-1)
        else:
            # 其他平台使用控制台蜂鸣声
            print('\a', end='', flush=True)
    except Exception as e:
        # 提示音失败不影响主程序
        logger.debug(f'Failed to play completion sound: {e}')

def safe_get_memory_info():
    """安全获取内存信息，失败时返回默认值"""
    try:
        memory = psutil.virtual_memory()
        return memory.percent, memory.available // (1024 * 1024)  # 可用内存MB
    except Exception as e:
        logger.warning(f'Unable to get memory info: {e}')
        return 95.0, 100  # 假设高内存使用率，低可用内存

def force_cleanup():
    """强制内存清理"""
    logger.debug('Performing force memory cleanup...')
    import gc
    import torch
    
    # Python垃圾回收    
    collected = gc.collect()
    
    # PyTorch缓存清理   
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 尝试清理更多内存  
    try:
        import ctypes
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except:
        pass

class MangaTranslatorLocal(MangaTranslator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.textlines = []
        self.attempts = params.get('attempts', None)
        self.skip_no_text = params.get('skip_no_text', False)
        self.text_output_file = params.get('text_output_file', None)
        self.save_quality = params.get('save_quality', None)
        self.text_regions = params.get('text_regions', None)
        self.save_text_file = params.get('save_text_file', None)
        self.save_text = params.get('save_text', None)
        self.prep_manual = params.get('prep_manual', None)
        self.batch_size = params.get('batch_size', 1)
        self.disable_memory_optimization = params.get('disable_memory_optimization', False)

    async def translate_path(self, path: str, dest: str = None, params: dict[str, Union[int, str]] = None):
        """
        Translates an image or folder (recursively) specified through the path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.abspath(os.path.expanduser(path))
        dest = os.path.abspath(os.path.expanduser(dest)) if dest else ''
        params = params or {}
        config_file_path = params.get("config_file", None)

        if config_file_path:
            try:
                with open(config_file_path, 'r') as file:
                    config_content = file.read()
            except Exception as e:
                print("Couldnt read file")
                raise e
            config_extension = os.path.splitext(config_file_path)[1].lower()

            try:
                if config_extension == ".toml":
                    import tomllib
                    config_dict = tomllib.loads(config_content)
                elif config_extension == ".json":
                    config_dict = json.loads(config_content)
                else:
                    raise ValueError("Unsupported configuration file format")
            except Exception as e:
                print("Failed to load configuration file")
                raise e
            config = Config(**config_dict)
        else:
            config = Config()
        # Handle format
        file_ext = params.get('format')
        if params.get('save_quality', 100) < 100:
            if not params.get('format'):
                file_ext = 'jpg'
            elif params.get('format') != 'jpg':
                raise ValueError('--save-quality of lower than 100 is only supported for .jpg files')

        if os.path.isfile(path):
            # Determine destination file path
            if not dest:
                # Use the same folder as the source
                p, ext = os.path.splitext(path)
                _dest = f'{p}-translated.{file_ext or ext[1:]}'
            elif not os.path.basename(dest):
                p, ext = os.path.splitext(os.path.basename(path))
                # If the folders differ use the original filename from the source
                if os.path.dirname(path) != dest:
                    _dest = os.path.join(dest, f'{p}.{file_ext or ext[1:]}')
                else:
                    _dest = os.path.join(dest, f'{p}-translated.{file_ext or ext[1:]}')
            else:
                p, ext = os.path.splitext(dest)
                _dest = f'{p}.{file_ext or ext[1:]}'
            await self.translate_file(path, _dest, params,config)

        elif os.path.isdir(path):
            # Determine destination folder path
            if path[-1] == '\\' or path[-1] == '/':
                path = path[:-1]
            _dest = dest or path + '-translated'
            if os.path.exists(_dest) and not os.path.isdir(_dest):
                raise FileExistsError(_dest)

            # 检查是否使用批量处理
            if self.batch_size > 1:
                await self._translate_folder_batch(path, _dest, params, config, file_ext)
            else:
                # 原有的逐个处理方式
                start_time = time.time()  # 记录开始时间
                translated_count = 0
                for root, subdirs, files in os.walk(path):
                    files = natural_sort(files)
                    dest_root = replace_prefix(root, path, _dest)
                    os.makedirs(dest_root, exist_ok=True)
                    for f in files:
                        if f.lower() == '.thumb':
                            continue

                        file_path = os.path.join(root, f)
                        output_dest = replace_prefix(file_path, path, _dest)
                        p, ext = os.path.splitext(output_dest)
                        output_dest = f'{p}.{file_ext or ext[1:]}'
                        try:
                            if await self.translate_file(file_path, output_dest, params, config):
                                translated_count += 1
                        except Exception as e:
                            logger.error(e)
                            raise e
                
                # 计算总耗时
                total_time = time.time() - start_time
                
                if translated_count == 0:
                    logger.info('No further untranslated files found. Use --overwrite to write over existing translations.')
                else:
                    # 格式化时间显示
                    if total_time >= 3600:  
                        time_str = f"{total_time/3600:.1f} hours"
                    elif total_time >= 60:  
                        time_str = f"{total_time/60:.1f} minutes"
                    else:  
                        time_str = f"{total_time:.1f} seconds"
                    
                    logger.info(f'Done. Translated {translated_count} image{"" if translated_count == 1 else "s"} in {time_str}')
                    logger.info(f'Results saved to: "{_dest}"')
                    try:
                        if ENABLE_COMPLETION_SOUND:
                            play_completion_sound()
                    except Exception as e:
                        logger.debug(f'Failed to play completion sound: {e}')

    async def translate_file(self, path: str, dest: str, params: dict, config: Config):
        if not params.get('overwrite') and os.path.exists(dest):
            logger.info(
                f'Skipping as already translated: "{dest}". Use --overwrite to overwrite existing translations.')
            await self._report_progress('saved', True)
            return True

        logger.info(f'Translating: "{path}"')

        # Turn dict to context to make values also accessible through params.<property>
        params = params or {}
        ctx = Context(**params)

        attempts = 0
        while self.attempts == -1 or attempts < self.attempts + 1:
            if attempts > 0:
                logger.info(f'Retrying translation! Attempt {attempts}'
                            + (f' of {self.attempts}' if self.attempts != -1 else ''))
            try:
                return await self._translate_file(path, dest, config, ctx)

            except TranslationInterrupt:
                break
            except Exception as e:
                if isinstance(e, LanguageUnsupportedException):
                    await self._report_progress('error-lang', True)
                else:
                    await self._report_progress('error', True)
                if not self.ignore_errors and not (self.attempts == -1 or attempts < self.attempts):
                    raise
                else:
                    logger.error(f'{e.__class__.__name__}: {e}',
                                 exc_info=e if self.verbose else None)
            attempts += 1
        return False

    async def _translate_file(self, path: str, dest: str, config: Config, ctx: Context) -> bool:
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                queries = f.read().splitlines()
            # 注入上下文参数到 ctx
            ctx.context_size = getattr(self, 'context_size', 0)
            ctx.all_page_translations = getattr(self, 'all_page_translations', [])
            translated_sentences = await dispatch_translation(
                config.translator.translator_gen,
                queries,
                config.translator,
                self.use_mtpe,
                ctx,
                device='cpu' if self._gpu_limited_memory else self.device
            )
            p, ext = os.path.splitext(dest)
            if ext != '.txt':
                dest = p + '.txt'
            logger.info(f'Saving "{dest}"')
            with open(dest, 'w') as f:
                f.write('\n'.join(translated_sentences))
            return True

        # TODO: Add .gif handler

        else:  # Treat as image
            try:
                img = Image.open(path)
                img.verify()
                img = Image.open(path)
            except Exception:
                logger.warn(f'Failed to open image: {path}')
                return False

            # 直接翻译图片，不再需要传递文件名
            ctx = await self.translate(img, config)
            result = ctx.result

            # TODO
            # Proper way to use the config but for now juste pass what we miss here ton ctx
            # Because old methods are still using for example ctx.gimp_font
            # Not done before because we change the ctx few lines above
            ctx.gimp_font = config.render.gimp_font

            # Save result
            if self.skip_no_text and not ctx.text_regions:
                logger.debug('Not saving due to --skip-no-text')
                return True
            if result:
                logger.info(f'Saving "{dest}"')
                ctx.save_quality = self.save_quality
                save_result(result, dest, ctx)
                await self._report_progress('saved', True)

                if self.save_text or self.save_text_file or self.prep_manual:
                    if self.prep_manual:
                        # Save original image next to translated
                        p, ext = os.path.splitext(dest)
                        img_filename = p + '-orig' + ext
                        img_path = os.path.join(os.path.dirname(dest), img_filename)
                        img.save(img_path, quality=self.save_quality)
                    if self.text_regions:
                        self._save_text_to_file(path, ctx)
                return True
        return False

    def _save_text_to_file(self, image_path: str, ctx: Context):
        cached_colors = []

        def identify_colors(fg_rgb: List[int]):
            idx = 0
            for rgb, _ in cached_colors:
                # If similar color already saved
                if abs(rgb[0] - fg_rgb[0]) + abs(rgb[1] - fg_rgb[1]) + abs(rgb[2] - fg_rgb[2]) < 50:
                    break
                else:
                    idx += 1
            else:
                cached_colors.append((fg_rgb, get_color_name(fg_rgb)))
            return idx + 1, cached_colors[idx][1]

        s = f'\n[{image_path}]\n'
        for i, region in enumerate(ctx.text_regions):
            fore, back = region.get_font_colors()
            color_id, color_name = identify_colors(fore)

            s += f'\n-- {i + 1} --\n'
            s += f'color: #{color_id}: {color_name} (fg, bg: {rgb2hex(*fore)} {rgb2hex(*back)})\n'
            s += f'text:  {region.text}\n'
            s += f'trans: {region.translation}\n'
            for line in region.lines:
                s += f'coords: {list(line.ravel())}\n'
        s += '\n'

        text_output_file = self.text_output_file
        if not text_output_file:
            text_output_file = os.path.splitext(image_path)[0] + '_translations.txt'

        with open(text_output_file, 'a', encoding='utf-8') as f:
            f.write(s)

    async def _translate_folder_batch(self, path: str, dest: str, params: dict, config: Config, file_ext: str):
        """使用批量处理方式翻译文件夹中的图片"""
        
        start_time = time.time()  # 记录开始时间
        memory_percent, available_mb = safe_get_memory_info()
        logger.info(f'Batch processing started - batch size: {self.batch_size}, memory usage: {memory_percent:.1f}%, available: {available_mb}MB')
        
        memory_optimization_enabled = not self.disable_memory_optimization
        if not memory_optimization_enabled:
            logger.info('Memory optimization disabled by user')
        else:
            logger.info('Memory optimization enabled')
        
        # 收集所有需要翻译的图片文件
        image_tasks = []
        for root, subdirs, files in os.walk(path):
            files = natural_sort(files)
            dest_root = replace_prefix(root, path, dest)
            os.makedirs(dest_root, exist_ok=True)
            
            for f in files:
                if f.lower() == '.thumb':
                    continue
                    
                file_path = os.path.join(root, f)
                output_dest = replace_prefix(file_path, path, dest)
                p, ext = os.path.splitext(output_dest)
                output_dest = f'{p}.{file_ext or ext[1:]}'
                
                # 检查是否需要跳过已翻译的文件
                if not params.get('overwrite') and os.path.exists(output_dest):
                    logger.debug(f'Skipping already translated file: "{output_dest}"')
                    continue
                    
                # 尝试加载图片
                try:
                    img = Image.open(file_path)
                    img.verify()
                    img = Image.open(file_path)  # 重新打开因为verify会关闭文件
                    image_tasks.append((img, config, file_path, output_dest))
                except Exception as e:
                    logger.warning(f'Failed to open image: {file_path}, error: {e}')
                    continue
        
        if not image_tasks:
            logger.info('No images found to translate, use --overwrite to write over existing translations.')
            return
            
        logger.info(f'Found {len(image_tasks)} images to translate')
        
        # 简化的内存优化策略
        base_batch_size = self.batch_size
        translated_count = 0
        i = 0
        
        while i < len(image_tasks):
            # 使用固定批次大小
            current_batch_size = base_batch_size
                
            batch = image_tasks[i:i + current_batch_size]
            batch_num = i // base_batch_size + 1
            total_batches = (len(image_tasks) + base_batch_size - 1) // base_batch_size
            
            logger.info(f'Processing batch {batch_num}/{total_batches} (size: {len(batch)})')
            
            # 内存状态检查
            memory_percent, available_mb = safe_get_memory_info()
            logger.debug(f'Memory status before batch: {memory_percent:.1f}%, available: {available_mb}MB')
            
            # 如果内存严重不足，强制清理
            if memory_optimization_enabled and memory_percent > 90:
                logger.warning(f'High memory usage detected ({memory_percent:.1f}%), forcing cleanup...')
                force_cleanup()
                memory_percent, available_mb = safe_get_memory_info()
                logger.info(f'Memory status after cleanup: {memory_percent:.1f}%, available: {available_mb}MB')
            
            # 创建当前批次的配置副本
            batch_config = config
            if memory_optimization_enabled:
                batch_config = copy.deepcopy(config)
                
                # 更新批次中的配置
                images_with_configs = [(img, batch_config) for img, _, _, _ in batch]
            else:
                images_with_configs = [(img, config) for img, _, _, _ in batch]
            
            try:
                # 批量翻译
                logger.debug(f'Starting batch translation for {len(batch)} images...')
                # 不再需要提取图片名称，直接进行批量翻译
                batch_results = await self.translate_batch(images_with_configs, len(batch))
                
                # 保存结果
                for j, (ctx, (img, _, file_path, output_dest)) in enumerate(zip(batch_results, batch)):
                    # 检查是否应该跳过没有文本的图片（遵循skip_no_text参数）
                    if self.skip_no_text and ctx and not ctx.text_regions:
                        logger.debug(f'Not saving due to --skip-no-text: {file_path}')
                        continue
                        
                    if ctx and ctx.result:
                        logger.debug(f'Saving translation result: "{output_dest}"')
                        save_ctx = Context(**params)
                        save_ctx.result = ctx.result
                        save_ctx.text_regions = ctx.text_regions
                        save_ctx.gimp_font = batch_config.render.gimp_font
                        save_ctx.save_quality = self.save_quality
                        
                        save_result(ctx.result, output_dest, save_ctx)
                        translated_count += 1
                        
                        # 保存文本文件（如果需要）
                        if self.save_text or self.save_text_file or self.prep_manual:
                            if self.prep_manual:
                                p, ext = os.path.splitext(output_dest)
                                img_filename = p + '-orig' + ext
                                img_path = os.path.join(os.path.dirname(output_dest), img_filename)
                                img.save(img_path, quality=self.save_quality)
                            if ctx.text_regions:
                                self._save_text_to_file(file_path, ctx)
                    else:
                        # 处理没有结果的情况 - 改进逻辑以区分不同情况
                        has_original_text = ctx and hasattr(ctx, 'text_regions') and ctx.text_regions
                        
                        if not ctx:
                            logger.warning(f'Translation failed: {file_path} (context is None)')
                            save_reason = "no_context"
                        elif not hasattr(ctx, 'result'):
                            logger.warning(f'Translation failed: {file_path} (no result attribute)')
                            save_reason = "no_result_attr"
                        elif ctx.result is None:
                            if has_original_text:
                                # 有原文但没有翻译结果，需要判断是否因为过滤导致
                                # 检查是否所有region都被过滤掉了（有translation但为空或被过滤）
                                filtered_by_processing = all(
                                    hasattr(region, 'translation') and 
                                    (not region.translation.strip() or  # 空翻译
                                     region.translation.isnumeric() or  # 数字翻译
                                     region.text.lower().strip() == region.translation.lower().strip())  # 翻译与原文相同
                                    for region in ctx.text_regions
                                ) if ctx.text_regions else False
                                
                                if filtered_by_processing:
                                    # logger.warning(f'Translation filtered out by post-processing: {file_path}')
                                    save_reason = "filtered_translation"
                                else:
                                    # logger.warning(f'Translation failed with original text present: {file_path} (result is None but has text_regions)')
                                    save_reason = "translation_failed_with_text"
                            else:
                                # logger.warning(f'Translation failed: {file_path} (result is None, no original text)')
                                save_reason = "no_original_text"
                        else:
                            logger.warning(f'Translation failed: {file_path} (unexpected condition)')
                            save_reason = "unexpected"
                            
                        # 决定是否保存图片
                        should_save = True
                        if save_reason == "translation_failed_with_text":
                            # 有原文但翻译失败且不是因为过滤导致，不保存图片以便重试
                            should_save = False
                            # logger.info(f'Skipping save for retry: {file_path} (translation failed but has original text)')
                        
                        # 如果不跳过无文本图片，且决定保存，则保存原图
                        if should_save and not self.skip_no_text:
                            logger.info(f'Saving original image ({save_reason}): {file_path}')
                            try:
                                # 确保目标目录存在
                                os.makedirs(os.path.dirname(output_dest), exist_ok=True)
                                
                                # 保存原图到目标位置
                                if self.save_quality and self.save_quality < 100:
                                    # 如果设置了压缩质量，转换为RGB并压缩保存
                                    img_copy = img.convert('RGB') if img.mode != 'RGB' else img.copy()
                                    img_copy.save(output_dest, quality=self.save_quality, format='JPEG')
                                else:
                                    # 保持原始格式和质量
                                    img.save(output_dest)
                                
                                logger.info(f'Original image saved: "{output_dest}"')
                                translated_count += 1  # 即使是原图也计入处理数量
                            except Exception as save_error:
                                logger.error(f'Failed to save original image: {file_path}, error: {save_error}')
                        else:
                            if not should_save:
                                logger.debug(f'Skipped saving for retry: {file_path}')
                            elif self.skip_no_text:
                                logger.debug(f'Skipped saving due to --skip-no-text: {file_path}')
                # 成功处理批次，重置连续错误计数
                logger.debug(f'Batch {batch_num} processed successfully')
                        
            except (MemoryError, OSError) as e:
                logger.error(f'Memory error in batch processing: {e}')
                
                if not memory_optimization_enabled:
                    logger.error('Consider enabling memory optimization (remove --disable-memory-optimization flag)')
                    raise
                
            except Exception as e:
                logger.error(f'Other error in batch processing: {e}')
                if not self.ignore_errors:
                    raise
                    
            # 清理当前批次资源
            for img, _, _, _ in batch:
                if hasattr(img, 'close'):
                    img.close()
                del img
            del batch
            
            # 每个批次后都执行内存清理
            force_cleanup()
            
            # 内存状态报告
            memory_percent, available_mb = safe_get_memory_info()
            logger.debug(f'Memory status after batch {batch_num}: {memory_percent:.1f}%, available: {available_mb}MB')
            
            # 移动到下一批次
            i += current_batch_size
            
        # 最终报告
        total_time = time.time() - start_time  # 计算总耗时
        
        if translated_count == 0:
            logger.info('No files to translate. Use --overwrite to overwrite existing translations.')
        else:
            # 格式化时间显示
            if total_time >= 3600:  
                time_str = f"{total_time/3600:.1f} hours"
            elif total_time >= 60:  
                time_str = f"{total_time/60:.1f} minutes"
            else:  
                time_str = f"{total_time:.1f} seconds"
            
            logger.info(f'Done! Translated {translated_count} image{"" if translated_count == 1 else "s"} in {time_str}')
            logger.info(f'Results saved to: "{dest}"')
            try:
                if ENABLE_COMPLETION_SOUND:
                    play_completion_sound()
            except Exception as e:
                logger.debug(f'Failed to play completion sound: {e}')
