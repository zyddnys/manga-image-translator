import asyncio
import logging
import regex as re
from typing import List, Optional, Tuple
import py3langid as langid

from .common import CommonTranslator, ISO_639_1_TO_VALID_LANGUAGES
from ..utils.generic import Context
from ..config import Config, Translator, TranslatorConfig
# Note: dispatch function is not needed in PostProcessorTranslator as it wraps existing translators

from ..utils.log import get_logger

logger = get_logger('PostProcessorTranslator')


class PostProcessorTranslator(CommonTranslator):
    """
    A wrapper translator that adds post-processing capabilities to any translator.
    包装器翻译器，为任意翻译器添加译后处理功能。
    
    This wrapper applies the following post-processing steps after translation:
    该包装器在翻译后应用以下译后处理步骤：
    1. Hallucination detection (幻觉检测)
    2. Target language validation with retry (目标语言验证与重试)
    3. Translation filtering (翻译过滤)
    4. Bracket consistency correction (括号一致性修正)
    """
    
    def __init__(self, inner_translator: CommonTranslator):
        """
        Initialize the post-processor wrapper.
        初始化译后处理包装器。
        
        Args:
            inner_translator: The translator to wrap
        """
        super().__init__()
        self.inner_translator = inner_translator
        self.config = None  # Will be set via parse_args
        
        # Copy language support from inner translator
        self._LANGUAGE_CODE_MAP = inner_translator._LANGUAGE_CODE_MAP
        self._INVALID_REPEAT_COUNT = inner_translator._INVALID_REPEAT_COUNT
        self._MAX_REQUESTS_PER_MINUTE = inner_translator._MAX_REQUESTS_PER_MINUTE
    
    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        """
        Delegate language support check to inner translator.
        将语言支持检查委托给内部翻译器。
        """
        return self.inner_translator.supports_languages(from_lang, to_lang, fatal)
    
    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        """
        Delegate language code parsing to inner translator.
        将语言代码解析委托给内部翻译器。
        """
        return self.inner_translator.parse_language_codes(from_lang, to_lang, fatal)
    
    def parse_args(self, config):
        """
        Parse configuration and pass it to inner translator.
        解析配置并传递给内部翻译器。
        """
        self.config = config
        if hasattr(self.inner_translator, 'parse_args'):
            self.inner_translator.parse_args(config)
    
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        """
        Delegate the actual translation to inner translator.
        将实际翻译委托给内部翻译器。
        """
        return await self.inner_translator._translate(from_lang, to_lang, queries)
    
    async def translate(self, from_lang: str, to_lang: str, queries: List[str], 
                       use_mtpe: bool = False, ctx: Optional[Context] = None) -> List[str]:
        """
        Translate queries and apply post-processing if Context is provided.
        翻译查询并在提供Context时应用译后处理。
        
        Args:
            from_lang: Source language
            to_lang: Target language  
            queries: List of text to translate
            use_mtpe: Whether to use machine translation post-editing
            ctx: Context object containing text_regions for post-processing
            
        Returns:
            List of translated and post-processed text
        """
        # First, get translations from inner translator
        # 首先，从内部翻译器获取翻译结果
        translations = await self.inner_translator.translate(from_lang, to_lang, queries, use_mtpe, ctx)
        
        # If we have Context with text_regions and config, apply post-processing
        # 如果有包含text_regions的Context和配置，应用译后处理
        if (ctx and hasattr(ctx, 'text_regions') and ctx.text_regions and
            self.config and hasattr(self.config, 'enable_post_translation_check') and
            self.config.enable_post_translation_check):

            # Check if this is batch mode with multiple contexts
            # 检查是否为包含多个上下文的批量模式
            is_batch_mode = (hasattr(ctx, 'batch_text_mapping') and 
                           hasattr(ctx, 'batch_contexts') and
                           ctx.batch_text_mapping is not None and
                           ctx.batch_contexts is not None)

            if is_batch_mode:
                # Batch mode: assign translations to all regions across all contexts
                # 批量模式：将翻译结果分配给所有上下文中的所有区域
                translation_idx = 0
                for ctx_idx, region_idx in ctx.batch_text_mapping:
                    target_ctx = ctx.batch_contexts[ctx_idx]
                    if region_idx < len(target_ctx.text_regions) and translation_idx < len(translations):
                        target_ctx.text_regions[region_idx].translation = translations[translation_idx]
                        target_ctx.text_regions[region_idx].target_lang = to_lang
                        translation_idx += 1
                    elif region_idx < len(target_ctx.text_regions):
                        # If we run out of translations, set empty translation
                        # 如果翻译结果不足，设置空翻译
                        target_ctx.text_regions[region_idx].translation = ""
                        target_ctx.text_regions[region_idx].target_lang = to_lang

                # Collect all text regions from all contexts for post-processing
                # 收集所有上下文中的文本区域用于译后处理
                all_regions = []
                for batch_ctx in ctx.batch_contexts:
                    if batch_ctx.text_regions:
                        all_regions.extend(batch_ctx.text_regions)

                # Temporarily replace ctx.text_regions with all regions for post-processing
                # 临时用所有区域替换ctx.text_regions进行译后处理
                original_regions = ctx.text_regions
                ctx.text_regions = all_regions
            else:
                # Single page mode: assign translations to current context regions
                # 单页模式：将翻译结果分配给当前上下文的区域
                translation_idx = 0
                for region in ctx.text_regions:
                    if translation_idx < len(translations):
                        region.translation = translations[translation_idx]
                        region.target_lang = to_lang
                        translation_idx += 1
                    else:
                        # If we run out of translations, set empty translation
                        # 如果翻译结果不足，设置空翻译
                        region.translation = ""
                        region.target_lang = to_lang
            
            # Apply post-processing steps in order
            # 按顺序应用译后处理步骤
            try:
                # Step 1: Hallucination detection (幻觉检测)
                await self._perform_hallucination_detection(ctx, self.config)
                
                # Step 2: Target language validation with retry (目标语言验证与重试)
                await self._perform_target_language_check_with_retry(ctx, self.config, "page")
                
                # Step 3: Translation filtering (翻译过滤)
                ctx.text_regions = await self._filter_translation_results(ctx, self.config)
                
                # Step 4: Bracket consistency correction (括号一致性修正)
                await self._apply_bracket_corrections(ctx)

                # Restore original regions if in batch mode
                # 如果是批量模式，恢复原始区域
                if is_batch_mode:
                    ctx.text_regions = original_regions
                    # Clean up batch mode attributes
                    # 清理批量模式属性
                    if hasattr(ctx, 'batch_text_mapping'):
                        delattr(ctx, 'batch_text_mapping')
                    if hasattr(ctx, 'batch_contexts'):
                        delattr(ctx, 'batch_contexts')

                # Extract final translations from processed regions
                # 从处理后的区域提取最终翻译结果
                final_translations = []
                for region in ctx.text_regions:
                    if hasattr(region, 'translation') and region.translation is not None:
                        final_translations.append(region.translation)
                    else:
                        final_translations.append("")

                # Update translations with processed results
                # 用处理后的结果更新翻译
                translations = final_translations
                
            except Exception as e:
                logger.error(f"Error during post-processing: {e}")
                # Clean up batch mode attributes if they exist
                # 如果存在批量模式属性，进行清理
                if hasattr(ctx, 'batch_text_mapping'):
                    delattr(ctx, 'batch_text_mapping')
                if hasattr(ctx, 'batch_contexts'):
                    delattr(ctx, 'batch_contexts')
                # If post-processing fails, return original translations
                # 如果译后处理失败，返回原始翻译结果
                pass
        
        return translations

    # Post-processing methods

    async def _perform_hallucination_detection(self, ctx: Context, config: Config):
        """
        执行幻觉检测并重试失败的区域 | Execute hallucination detection and retry failed regions
        """
        # 检查text_regions是否为None或空 | Check if text_regions is None or empty
        if not ctx.text_regions:
            return

        # 单个region幻觉检测 | Single region hallucination detection
        failed_regions = []
        if config.enable_post_translation_check:
            region_count = len(ctx.text_regions) if ctx.text_regions else 0
            logger.info(f"Starting hallucination detection with {region_count} regions...")

            # 单个region级别的幻觉检测 | Single region level hallucination detection
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 检查重复内容幻觉 | Check repetition hallucination
                    if await self._check_repetition_hallucination(
                        region.translation,
                        config.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)

            # 对失败的区域进行重试 | Retry failed regions
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed hallucination check, starting retry...")
                for region in failed_regions:
                    try:
                        old_translation = region.translation  # 保存原始翻译
                        logger.info(f"Retrying translation for region with text: '{region.text}'")
                        new_translation = await self._retry_single_region_translation(region, config, ctx)

                        # 检查重试是否成功（翻译内容发生变化且通过了幻觉检测）
                        if new_translation != old_translation:
                            logger.info(f"Region retry successful: '{region.text}' -> '{new_translation}'")
                        else:
                            # 重试失败，显示失败日志
                            logger.warning(f'Post-translation check failed after all {config.post_check_max_retry_attempts} retries, keeping original: "{old_translation}"')
                    except Exception as e:
                        logger.error(f"Error during region retry: {e}")

    async def _retry_single_region_translation(self, region, config: Config, ctx: Context) -> str:
        """
        重试单个区域的翻译直到通过幻觉检测 | Retry translation of single region until passing hallucination detection
        """
        original_translation = region.translation
        max_attempts = config.post_check_max_retry_attempts

        # 第一次检查原始翻译（不算重试）+ max_attempts次重试
        for attempt in range(max_attempts + 1):
            # 检查当前翻译是否有幻觉 | Check if current translation has hallucination
            has_hallucination = await self._check_repetition_hallucination(
                region.translation,
                config.post_check_repetition_threshold,
                silent=True  # 重试过程中禁用日志输出 | Disable logging during retry
            )

            if not has_hallucination:
                if attempt > 0:
                    logger.info(f'Hallucination check passed: "{region.translation}"')
                return region.translation

            # 如果不是最后一次尝试，进行重新翻译 | If not the last attempt, perform re-translation
            if attempt < max_attempts:
                if attempt == 0:
                    logger.warning(f'Post-translation check failed for original translation, starting retry (Retry 1/{max_attempts}): "{region.text}"')
                else:
                    logger.warning(f'Post-translation check failed (Retry {attempt}/{max_attempts}), re-translating: "{region.text}"')

                try:
                    # 单独重新翻译这个文本区域 | Re-translate this text region individually
                    if config.translator != Translator.none:
                        # 直接使用内部翻译器重新翻译 | Use inner translator directly for re-translation
                        retranslated = await self.inner_translator.translate(
                            'auto',
                            config.target_lang,
                            [region.text],
                            getattr(self, 'use_mtpe', False),
                            ctx
                        )
                        if retranslated:
                            region.translation = retranslated[0]

                            # 应用格式化处理 | Apply formatting
                            if hasattr(config, 'render') and config.render.uppercase:
                                region.translation = region.translation.upper()
                            elif hasattr(config, 'render') and config.render.lowercase:
                                region.translation = region.translation.lower()
                    else:
                        # 如果翻译器是none，直接返回原文
                        region.translation = region.text

                except Exception as e:
                    logger.error(f"Error during region retry {attempt if attempt > 0 else 1}: {e}")
                    break
            else:
                # 最后一次重试失败，恢复原始翻译
                region.translation = original_translation

        # 返回最终的翻译结果（可能是原始翻译或重试后的翻译）
        return region.translation

    async def _check_repetition_hallucination(self, text: str, threshold: int = 5, silent: bool = False) -> bool:
        """
        检查文本是否包含重复内容（模型幻觉） | Check if the text contains repetitive content (model hallucination)
        """
        if not text or len(text.strip()) < threshold:
            return False

        # 检查字符级重复 | Check character-level repetition
        consecutive_count = 1
        max_consecutive_count = 1
        prev_char = None
        hallucination_detected = False

        for char in text:
            if char == prev_char:
                consecutive_count += 1
                max_consecutive_count = max(max_consecutive_count, consecutive_count)
                if consecutive_count >= threshold and not hallucination_detected:
                    hallucination_detected = True
            else:
                consecutive_count = 1
            prev_char = char

        if hallucination_detected:
            if not silent:
                logger.warning(f'Detected character repetition hallucination: "{text}" - repeated character: "{prev_char}", consecutive count: {max_consecutive_count}')
            return True

        # 检查词语级重复（按字符分割中文，按空格分割其他语言） | Check word-level repetition (split Chinese by character, other languages by space)
        segments = re.findall(r'[\u4e00-\u9fff]|\S+', text)

        if len(segments) >= threshold:
            consecutive_segments = 1
            max_consecutive_segments = 1
            prev_segment = None
            word_hallucination_detected = False

            for segment in segments:
                if segment == prev_segment:
                    consecutive_segments += 1
                    max_consecutive_segments = max(max_consecutive_segments, consecutive_segments)
                    if consecutive_segments >= threshold and not word_hallucination_detected:
                        word_hallucination_detected = True
                else:
                    consecutive_segments = 1
                prev_segment = segment

            if word_hallucination_detected:
                if not silent:
                    logger.warning(f'Detected word repetition hallucination: "{text}" - repeated segment: "{prev_segment}", consecutive count: {max_consecutive_segments}')
                return True

        # 检查词语重复模式 | Check word pattern repetition
        # 检测文本开头的2-4字符重复模式 | Detect 2-4 character repetition patterns at the beginning of text
        for pattern_length in [2, 3, 4]:
            if pattern_length * 3 > len(text):  # 至少需要3次重复才有意义 | At least 3 repetitions needed to be meaningful
                continue

            pattern = text[:pattern_length]
            # 快速检查：如果文本开头的模式重复出现 | Quick check: if the pattern at the beginning repeats
            if len(text) >= pattern_length * 3 and text.startswith(pattern * 3):
                repeat_count = 0
                pos = 0
                # 计算连续重复次数 | Count consecutive repetitions
                while pos + pattern_length <= len(text) and text[pos:pos + pattern_length] == pattern:
                    repeat_count += 1
                    pos += pattern_length

                # 如果重复次数达到阈值 | If repetition count reaches threshold
                if repeat_count >= max(3, threshold // 2):
                    if not silent:
                        logger.warning(f'Detected pattern repetition hallucination: "{text}" - repeated pattern: "{pattern}", repeat count: {repeat_count}')
                    return True

        # 检查短语级重复 | Check phrase-level repetition
        words = text.split()
        if len(words) >= 6:  # 至少6个单词才检查短语重复 | At least 6 words needed to check phrase repetition
            # 检查2-3个单词的短语重复 | Check 2-3 word phrase repetition
            for phrase_len in [2, 3]:
                if phrase_len * 3 > len(words):
                    continue

                # 检查开头短语的重复 | Check repetition of the beginning phrase
                phrase = ' '.join(words[:phrase_len])
                if phrase and len(phrase.strip()) > 1:  # 确保短语有意义 | Ensure phrase is meaningful
                    # 计算这个短语在文本中出现的次数 | Count occurrences of this phrase in the text
                    phrase_count = text.count(phrase)
                    if phrase_count >= 3:  # 至少出现3次 | At least 3 occurrences
                        if not silent:
                            logger.warning(f'Detected phrase repetition hallucination: "{text}" - repeated phrase: "{phrase}", occurrence count: {phrase_count}')
                        return True

        return False

    async def _check_target_language_ratio(self, text_regions: List, target_lang: str, min_ratio: float = 0.5, threshold: int = 0) -> bool:
        """
        检查翻译结果中目标语言的占比是否达到要求 | Check if the target language ratio meets the requirement
        使用py3langid进行语言检测 | Use py3langid for language detection

        Args:
            text_regions: 文本区域列表 | List of text regions
            target_lang: 目标语言代码 | Target language code
            min_ratio: 最小目标语言占比（此参数在新逻辑中不使用，保留为兼容性） | Minimum target language ratio (not used in new logic, kept for compatibility)
            threshold: 最小区域数量阈值，如果区域数量 <= threshold 则跳过检查 | Minimum region count threshold, skip check if region count <= threshold

        Returns:
            bool: True表示通过检查，False表示未通过 | True means passed check, False means failed
        """
        if not text_regions or (threshold > 0 and len(text_regions) <= threshold):
            # 如果区域数量不超过阈值，跳过此检查 | If region count doesn't exceed threshold, skip this check
            return True

        # 合并所有翻译文本 | Merge all translation texts
        all_translations = []
        for region in text_regions:
            translation = getattr(region, 'translation', '')
            if translation and translation.strip():
                all_translations.append(translation.strip())

        if not all_translations:
            logger.debug('No valid translation texts for language ratio check')
            return True

        # 将所有翻译合并为一个文本进行检测 | Merge all translations into one text for detection
        merged_text = ''.join(all_translations)

        # 使用py3langid进行语言检测 | Use py3langid for language detection
        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
            if detected_language != 'UNKNOWN':
                detected_language = detected_language.upper()

        except Exception as e:
            logger.debug(f'py3langid failed for merged text: {e}')
            detected_language = 'UNKNOWN'
            confidence = -9999

        # 检查检测出的语言是否为目标语言 | Check if detected language matches target language
        is_target_lang = (detected_language == target_lang.upper())

        return is_target_lang

    async def _perform_target_language_check_with_retry(self, contexts_or_ctx, config: Config = None, check_type: str = "page"):
        """
        统一的目标语言检查和重试逻辑 | Unified target language check and retry logic

        Args:
            contexts_or_ctx: 单个Context或Context列表（批量模式） | Single Context or Context list (batch mode)
            config: 配置对象（批量模式时可为None，会从第一个Context获取） | Configuration object (can be None in batch mode, will get from first Context)
            check_type: 检查类型 ("page", "single", "batch") | Check type ("page", "single", "batch")
        """
        # 处理不同的输入类型 | Handle different input types
        if check_type == "batch":
            batch = contexts_or_ctx
            if not batch or not batch[0][1].translator.enable_post_translation_check:
                return
            config = batch[0][1]

            # 收集所有regions | Collect all regions
            all_regions = []
            for ctx, _ in batch:
                if ctx.text_regions:
                    all_regions.extend(ctx.text_regions)

            min_ratio = config.post_check_target_lang_threshold
            threshold = 10  # 批量模式保持较高阈值 | Batch mode maintains higher threshold
            check_name = "batch-level"
        else:
            # 单个Context的情况 | Single Context case
            ctx = contexts_or_ctx
            if not config.enable_post_translation_check or not ctx.text_regions:
                return

            all_regions = ctx.text_regions

            if check_type == "single":
                min_ratio = config.post_check_target_lang_threshold
                threshold = 3  # 统一阈值为3 | Unified threshold to 3
                check_name = "single image"
            else:  # page
                min_ratio = config.post_check_target_lang_threshold
                threshold = 3  # 统一阈值为3 | Unified threshold to 3
                check_name = "page-level"

        # 检查是否需要进行目标语言检查 | Check if target language check is needed
        if threshold > 0 and len(all_regions) <= threshold:
            logger.info(f"Skipping {check_name} target language check: only {len(all_regions)} regions (threshold: {threshold})")
            return

        # 执行目标语言检查 | Execute target language check
        logger.info(f"Starting {check_name} target language check with {len(all_regions)} regions...")
        lang_check_result = await self._check_target_language_ratio(
            all_regions,
            config.target_lang,
            min_ratio=min_ratio,
            threshold=threshold
        )

        if not lang_check_result:
            logger.warning(f"{check_name.capitalize()} target language ratio check failed")

            # 重试逻辑 | Retry logic
            max_retry = config.post_check_max_retry_attempts
            retry_count = 0

            while retry_count < max_retry and not lang_check_result:
                retry_count += 1
                logger.warning(f"Starting {check_name} retry {retry_count}/{max_retry}")

                # 根据类型执行不同的重试逻辑 | Execute different retry logic based on type
                if check_type == "batch":
                    lang_check_result = await self._retry_translation_batch(batch, config, retry_count, min_ratio, is_batch=True)
                else:
                    lang_check_result = await self._retry_translation_batch(ctx, config, retry_count, min_ratio, is_batch=False)

                if lang_check_result:
                    logger.info(f"{check_name.capitalize()} target language check passed after retry {retry_count}")
                    break

            if not lang_check_result:
                logger.error(f"{check_name.capitalize()} target language check failed after all {max_retry} retries")
        else:
            logger.info(f"{check_name.capitalize()} target language ratio check passed")

        # 统一的成功信息 | Unified success message
        if lang_check_result:
            logger.info("All translation regions passed post-translation check.")
        else:
            logger.warning("Some translation regions failed post-translation check.")

    async def _retry_translation_batch(self, contexts_or_ctx, config: Config, retry_count: int, min_ratio: float, is_batch: bool = False) -> bool:
        """统一的重试逻辑 | Unified retry logic"""
        if is_batch:
            # 批量模式：处理多个Context | Batch mode: handle multiple Contexts
            batch = contexts_or_ctx
            all_original_texts = []
            region_mapping = []  # 记录每个text属于哪个ctx | Record which ctx each text belongs to

            for ctx_idx, (ctx, _) in enumerate(batch):
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        if hasattr(region, 'text') and region.text:
                            all_original_texts.append(region.text)
                            region_mapping.append((ctx_idx, region))

            if not all_original_texts:
                logger.warning("No text found for batch retry")
                return False

            try:
                # 重新批量翻译 | Re-translate in batch
                logger.info(f"Retrying translation for {len(all_original_texts)} regions...")
                new_translations = await self._batch_translate_texts(all_original_texts, config, batch[0][0])

                # 更新翻译结果到各个region | Update translation results to each region
                for i, (ctx_idx, region) in enumerate(region_mapping):
                    if i < len(new_translations) and new_translations[i]:
                        old_translation = region.translation
                        region.translation = new_translations[i]
                        logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")

                # 重试后需要重新进行幻觉检测 | Need to re-run hallucination detection after retry
                for ctx, config_item in batch:
                    if ctx.text_regions:
                        await self._perform_hallucination_detection(ctx, config_item)

                # 重新收集所有regions并检查目标语言比例 | Re-collect all regions and check target language ratio
                all_batch_regions = []
                for ctx, _ in batch:
                    if ctx.text_regions:
                        all_batch_regions.extend(ctx.text_regions)

                logger.info(f"Re-checking batch-level target language ratio after batch retry {retry_count}...")
                return await self._check_target_language_ratio(
                    all_batch_regions,
                    config.target_lang,
                    min_ratio=min_ratio,
                    threshold=10  # 批量模式使用阈值10 | Batch mode uses threshold 10
                )

            except Exception as e:
                logger.error(f"Error during batch retry {retry_count}: {e}")
                return False
        else:
            # 单页模式：处理单个Context | Single page mode: handle single Context
            ctx = contexts_or_ctx
            original_texts = [region.text for region in ctx.text_regions if hasattr(region, 'text') and region.text]
            if not original_texts:
                return False

            try:
                new_translations = await self._batch_translate_texts(original_texts, config, ctx)

                # 更新翻译结果 | Update translation results
                text_idx = 0
                for region in ctx.text_regions:
                    if hasattr(region, 'text') and region.text and text_idx < len(new_translations):
                        old_translation = region.translation
                        region.translation = new_translations[text_idx]
                        logger.debug(f"Region translation updated: '{old_translation}' -> '{new_translations[text_idx]}'")
                        text_idx += 1

                # 重试后需要重新进行幻觉检测 | Need to re-run hallucination detection after retry
                await self._perform_hallucination_detection(ctx, config)

                # 重新检查目标语言比例 | Re-check target language ratio
                return await self._check_target_language_ratio(
                    ctx.text_regions,
                    config.target_lang,
                    min_ratio=min_ratio,
                    threshold=3  # 单页模式使用阈值3 | Single page mode uses threshold 3
                )

            except Exception as e:
                logger.error(f"Error during single retry {retry_count}: {e}")
                return False

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context) -> List[str]:
        """
        批量翻译文本的辅助方法 | Helper method for batch translating texts
        """
        try:
            # 直接使用内部翻译器进行批量翻译 | Use inner translator directly for batch translation
            return await self.inner_translator.translate(
                'auto',
                config.target_lang,
                texts,
                getattr(self, 'use_mtpe', False),
                ctx
            )
        except Exception as e:
            logger.error(f"Error during batch translation: {e}")
            return []

    async def _filter_translation_results(self, ctx: Context, config: Config) -> List:
        """
        过滤翻译结果，移除不需要的翻译 | Filter translation results, remove unwanted translations
        """
        if not ctx.text_regions:
            return []

        new_text_regions = []
        for region in ctx.text_regions:
            should_filter = False
            filter_reason = ""

            if not region.translation.strip():
                should_filter = True
                filter_reason = "Translation contain blank areas"
            elif config.translator != Translator.none:
                if region.translation.isnumeric():
                    should_filter = True
                    filter_reason = "Numeric translation"
                elif hasattr(config, 'filter_text') and config.filter_text and hasattr(config, 're_filter_text') and re.search(config.re_filter_text, region.translation):
                    should_filter = True
                    filter_reason = f"Matched filter text: {config.filter_text}"
                elif not config.translator == Translator.original:
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

    async def _apply_bracket_corrections(self, ctx: Context):
        """
        保持括号一致性 | Maintain bracket consistency
        """
        if not ctx.text_regions:
            return

        check_items = [
            # 圆括号处理 | Round bracket handling
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],

            # 方括号处理 | Square bracket handling
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],

            # 引号处理 | Quote handling
            ["「", """, "'", "『", "【"],
            ["」", """, "'", "』", "】"],
            ["『", """, "'", "「", "【"],
            ["』", """, "'", "」", "】"],
            # 新增【】处理 | Added 【】 handling
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", """],
            ["「", "'"],
            ["」", """],
            ["」", "'"],
            ["【", "["],
            ["】", "]"],
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                # 引号处理逻辑 | Quote handling logic
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text:
                    quote_type = '【】'
                else:
                    quote_type = None

                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')

                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):

                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # 括号修正逻辑 | Bracket correction logic
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])

                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 强制替换规则 | Forced replacement rules
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

    # Delegate other methods to inner translator if they exist
    # 如果内部翻译器有其他方法，则委托给它们
    def __getattr__(self, name):
        """
        Delegate any missing attributes/methods to inner translator.
        将任何缺失的属性/方法委托给内部翻译器。
        """
        return getattr(self.inner_translator, name)
