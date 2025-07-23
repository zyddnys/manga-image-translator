import os, re, asyncio, base64, json
from io import BytesIO
from typing import List
from collections import Counter
from loguru import logger
from PIL import Image
from manga_translator.utils import is_valuable_text
from .chatgpt import OpenAITranslator
from ..utils import Context
from .keys import OPENAI_API_KEY, OPENAI_MODEL


def encode_image(image):
    max_dim = 1024
    w, h = image.size
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode('utf-8'), new_w, new_h


class RefusalMessageError(Exception):
    """Raised when the LMM returns a refusal message instead of JSON."""
    pass


class ChatGPT2StageTranslator(OpenAITranslator):
    """
    ChatGPT three-stage translator with text reordering:
    Stage 1: Use ChatGPT vision to correct OCR errors and reorder text regions by reading sequence
    Stage 2: Translate the reordered text using corrected reading sequence
    Stage 3: Remap translations back to original positions to maintain correct placement
    Maintains all functionality from the base ChatGPT translator including glossary support, retry mechanisms, etc.
    """
    
    # RPM速率限制 - 防止429错误
    _MAX_REQUESTS_PER_MINUTE = 15  # 每分钟最大请求数，可根据API限制调整
    
    # JSON Schema for structured output (single image)
    REFINE_RESPONSE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "ocr_refinement_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "corrected_regions": {
                        "type": "array",
                        "description": "Array of text regions with corrected OCR and reading order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reading_order": {
                                    "type": "integer",
                                    "description": "The correct reading order index for this text region"
                                },
                                "original_bbox_id": {
                                    "type": "integer",
                                    "description": "The original bounding box ID from the input"
                                },
                                "bbox_2d": {
                                    "type": "array",
                                    "description": "Bounding box coordinates as [x1, y1, x2, y2]",
                                    "items": {"type": "integer"}
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Original OCR text"
                                },
                                "corrected_text": {
                                    "type": "string",
                                    "description": "OCR-corrected text"
                                }
                            },
                            "required": ["reading_order", "original_bbox_id", "bbox_2d", "text", "corrected_text"],
                            "additionalProperties": False
                        }
                    },
                    "image_received": {
                        "type": "boolean",
                        "description": "Confirmation that the image was received and processed"
                    }
                },
                "required": ["corrected_regions", "image_received"],
                "additionalProperties": False
            }
        }
    }

    # JSON Schema for batch structured output
    BATCH_REFINE_RESPONSE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "batch_ocr_refinement_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "batch_results": {
                        "type": "array",
                        "description": "Array of results for each image in the batch",
                        "items": {
                            "type": "object",
                            "properties": {
                                "image_index": {
                                    "type": "integer",
                                    "description": "Index of the image in the batch (0-based)"
                                },
                                "corrected_regions": {
                                    "type": "array",
                                    "description": "Array of text regions with corrected OCR and reading order for this image",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "reading_order": {
                                                "type": "integer",
                                                "description": "The correct reading order index within this image"
                                            },
                                            "original_bbox_id": {
                                                "type": "integer",
                                                "description": "The original bounding box ID from the input"
                                            },
                                            "bbox_2d": {
                                                "type": "array",
                                                "description": "Bounding box coordinates as [x1, y1, x2, y2]",
                                                "items": {"type": "integer"}
                                            },
                                            "text": {
                                                "type": "string",
                                                "description": "Original OCR text"
                                            },
                                            "corrected_text": {
                                                "type": "string",
                                                "description": "OCR-corrected text"
                                            }
                                        },
                                        "required": ["reading_order", "original_bbox_id", "bbox_2d", "text", "corrected_text"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["image_index", "corrected_regions"],
                            "additionalProperties": False
                        }
                    },
                    "images_received": {
                        "type": "integer",
                        "description": "Number of images that were received and processed in this batch"
                    }
                },
                "required": ["batch_results", "images_received"],
                "additionalProperties": False
            }
        }
    }
    
    _LANG_PATTERNS = [
        ('JPN', r'[\u3040-\u309f\u30a0-\u30ff]'),
        ('KOR', r'[\uac00-\ud7af\u1100-\u11ff]'),
        ('CN', r'[\u4e00-\u9fff]'),
        ('ARA', r'[\u0600-\u06ff]'),
        ('THA', r'[\u0e00-\u0e7f]'),
        ('RUS', r'[\u0400-\u04ff]')
    ]
    _LEFT_SYMBOLS = ['(', '（', '[', '【', '{', '〔', '〈', '「', '"', "'", '《', '『', '"', '〝', '﹁', '﹃', '⸂', '⸄', '⸉', '⸌',
                     '⸜', '⸠', '‹', '«']
    _RIGHT_SYMBOLS = [')', '）', ']', '】', '}', '〕', '〉', '」', '"', "'", '》', '』', '"', '〞', '﹂', '﹄', '⸃', '⸅', '⸊',
                      '⸍', '⸝', '⸡', '›', '»']

    # 拒绝回应检测关键词（正则）
    KEYWORDS = [
        r"I must decline",
        r"(i('m| am)?\s+)?sorry(.|\n)*?(can(['’]t|not)|unable to)\s+(assist|help)",
        r"unable to (assist|help)",
        r"cannot (assist|help)",
        r"(抱歉，|对不起，)",
        r"我(无法[将把]|不[能会便](提供|处理))",
        r"我无法(满足|回答|处理|提供)",
        r"这超出了我的范围",
        r"我需要婉拒",
        r"翻译或生成",
        r"[个]内容(吧)?",
        r"申し訳ありませんが",
    ]

    @classmethod
    def _contains_refusal(cls, text: str) -> bool:
        """Check whether the returned text contains a refusal message."""
        for pattern in cls.KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    async def _attempt_fallback_stage1(self, refine_prompt: str, base64_img: str, from_lang: str, queries: List[str]):
        """统一的 Stage-1 fallback 逻辑，避免在多处重复代码。"""
        if not hasattr(self, "_fallback_model") or not self._fallback_model:
            self.logger.debug("No fallback model configured, keeping original texts.")
            return queries, list(range(len(queries)))

        fallback_max_attempts = 3
        for fb_attempt in range(fallback_max_attempts):
            self.logger.warning(
                f"Trying fallback model '{self._fallback_model}' for Stage 1 OCR (attempt {fb_attempt+1}/{fallback_max_attempts})")
            try:
                await self._ratelimit_sleep()
                response_fb = await self.client.chat.completions.create(
                    model=self._fallback_model,
                    messages=[
                        {"role": "system", "content": self._get_refine_system_instruction(from_lang)},
                        {"role": "user", "content": [
                            {"type": "text", "text": refine_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]}
                    ],
                    temperature=self.refine_temperature,
                    max_completion_tokens=self.max_tokens,
                    response_format=self.REFINE_RESPONSE_SCHEMA,
                )

                if response_fb and response_fb.choices and response_fb.choices[0].message.content:
                    raw_content_fb = response_fb.choices[0].message.content

                    # 如果回退模型仍拒绝，则直接退出 / still refusal -> abort
                    if self._contains_refusal(raw_content_fb):
                        self.logger.warning(f"Fallback model also refused: '{raw_content_fb}'. Using original texts.")
                        break

                    return self._parse_json_response(raw_content_fb, queries)
                else:
                    self.logger.warning(f"Fallback Stage1 OCR attempt {fb_attempt+1}/{fallback_max_attempts} failed: Received empty response from model.")

            except Exception as fb_err:
                self.logger.warning(
                    f"Fallback Stage1 OCR attempt {fb_attempt+1}/{fallback_max_attempts} failed: {fb_err}")
                if fb_attempt < fallback_max_attempts - 1:
                    await asyncio.sleep(1)

        # 所有回退尝试失败 / All fallback attempts failed
        self.logger.warning("All Stage 1 fallback attempts failed. Proceeding to Stage 2 with original texts.")
        return queries, list(range(len(queries)))

    async def _attempt_batch_fallback_stage1(self, batch_refine_prompt: str, batch_base64_images: List[str],
                                           from_lang: str, queries: List[str], query_to_image_mapping: List[tuple]):
        """批量 Stage-1 fallback 逻辑，在一个请求中处理多张图片。"""
        if not hasattr(self, "_fallback_model") or not self._fallback_model:
            self.logger.debug("No fallback model configured for batch processing, keeping original texts.")
            return queries, list(range(len(queries)))

        fallback_max_attempts = 3
        for fb_attempt in range(fallback_max_attempts):
            self.logger.warning(
                f"Trying batch fallback model '{self._fallback_model}' for Stage 1 OCR (attempt {fb_attempt+1}/{fallback_max_attempts})")
            try:
                await self._ratelimit_sleep()

                # Construct messages with multiple images for fallback
                user_content = [{"type": "text", "text": batch_refine_prompt}]
                for base64_img in batch_base64_images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })

                response_fb = await self.client.chat.completions.create(
                    model=self._fallback_model,
                    messages=[
                        {"role": "system", "content": self._get_batch_refine_system_instruction(from_lang)},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=self.refine_temperature,
                    max_completion_tokens=self.max_tokens,
                    response_format=self.BATCH_REFINE_RESPONSE_SCHEMA,
                )

                if response_fb and response_fb.choices and response_fb.choices[0].message.content:
                    raw_content_fb = response_fb.choices[0].message.content

                    # 如果回退模型仍拒绝，则直接退出
                    if self._contains_refusal(raw_content_fb):
                        self.logger.warning(f"Batch fallback model also refused: '{raw_content_fb}'. Using original texts.")
                        break

                    # 解析批量响应
                    batch_reordered_texts, batch_position_mapping = self._parse_batch_json_response(
                        raw_content_fb, queries, query_to_image_mapping
                    )

                    self.logger.info(f"Batch fallback model succeeded: {len(batch_reordered_texts)} texts reordered")
                    return batch_reordered_texts, batch_position_mapping
                else:
                    self.logger.warning(f"Batch fallback Stage1 OCR attempt {fb_attempt+1}/{fallback_max_attempts} failed: Received empty response from model.")

            except Exception as fb_err:
                self.logger.warning(
                    f"Batch fallback Stage1 OCR attempt {fb_attempt+1}/{fallback_max_attempts} failed: {fb_err}")
                if fb_attempt < fallback_max_attempts - 1:
                    await asyncio.sleep(1)

        # 所有批量回退尝试失败
        self.logger.warning("All batch Stage 1 fallback attempts failed. Proceeding to Stage 2 with original texts.")
        return queries, list(range(len(queries)))

    def __init__(self, max_tokens=16000, refine_temperature=0.0, translate_temperature=0.1, stage1_retry_count=2, stage2_send_image=True, stage1_model=None, stage2_model=None):
        super().__init__()
        self.max_tokens = max_tokens
        self.refine_temperature = refine_temperature
        self.translate_temperature = translate_temperature
        self.stage1_retry_count = stage1_retry_count  # 添加Stage1重试次数参数
        self.stage2_send_image = stage2_send_image     # 控制Stage2是否发送图片
        
        # 双模型配置 - 支持环境变量配置
        self.stage1_model = stage1_model or os.getenv('OPENAI_STAGE1_MODEL') or OPENAI_MODEL
        self.stage2_model = stage2_model or os.getenv('OPENAI_STAGE2_MODEL') or OPENAI_MODEL
        
        # 添加第二阶段翻译标志位和图片存储
        self._is_stage2_translation = False
        self._stage2_image_base64 = None
        self._stage2_use_fallback = False  # 新增：Stage2回退模型激活标志
        
        # Check model configuration and warn once
        if not hasattr(ChatGPT2StageTranslator, '_warned_about_model'):
            self.logger.warning("⚠️ ChatGPT2Stage requires Large Multimodal Models (LMMs) for Stage 1 OCR correction!")
            if self.stage1_model == self.stage2_model:
                self.logger.info(f"Using single model for both stages: {self.stage1_model}")
            else:
                self.logger.info(f"Using dual models - Stage 1: {self.stage1_model}, Stage 2: {self.stage2_model}")
            ChatGPT2StageTranslator._warned_about_model = True

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx: Context = None) -> List[str]:
        """
        Override the base translate method to implement 2-stage translation
        """
        if not queries:
            return queries

        if ctx is None:
            self.logger.warning("No context provided, falling back to single-stage translation")
            return await super()._translate(from_lang, to_lang, queries)

        # Check if this is a batch processing scenario
        batch_contexts = getattr(ctx, 'batch_contexts', None)
        if batch_contexts and len(batch_contexts) > 1:
            # Batch processing mode
            return await self._translate_batch_2stage(from_lang, to_lang, queries, batch_contexts)
        else:
            # Single image processing mode
            return await self._translate_2stage(from_lang, to_lang, queries, ctx)

    async def _translate_2stage(self, from_lang: str, to_lang: str, queries: List[str], ctx: Context) -> List[str]:
        """
        Three-stage translation process with text reordering:
        1. Stage 1: OCR correction and text region reordering by reading sequence
        2. Stage 2: Translation using reordered text 
        3. Stage 3: Remap translations back to original positions
        """
        try:
            # Get RGB image and text regions
            rgb_img = Image.fromarray(ctx.img_rgb)
            w, h = rgb_img.size
            
            # Use all text regions directly, maintaining original order
            query_regions = ctx.text_regions[:len(queries)] if ctx.text_regions else []
            
            # Pad with None if we have more queries than regions
            while len(query_regions) < len(queries):
                query_regions.append(None)
                
            # Log region info for debugging
            self.logger.debug(f"Processing {len(queries)} queries with {len(ctx.text_regions)} text regions")
            self.logger.debug(f"Original query order: {queries}")

            # Stage 1: OCR correction and text reordering
            self.logger.info(f"Stage 1: Correcting OCR errors and reordering text regions using {self.stage1_model}...")
            base64_img, nw, nh = encode_image(rgb_img)
            refine_prompt = self._get_refine_prompt(query_regions, w, h, nw, nh)

            # Log the JSON content being sent to OCR model
            self.logger.info("Stage 1 OCR Request - JSON Content:")
            self.logger.info(f"{refine_prompt}")

            # 默认回退值，若随后成功解析将被覆盖
            reordered_texts = queries
            original_position_mapping = list(range(len(queries)))

            response = None
            for retry_count in range(self.stage1_retry_count + 1): # +1 for the initial try
                try:
                    # RPM速率限制 - 防止429错误
                    await self._ratelimit_sleep()
                    
                    # Use structured output for reliable JSON formatting
                    response = await self.client.chat.completions.create(
                        model=self.stage1_model,  # Use specified Stage 1 model
                        messages=[
                            {"role": "system", "content": self._get_refine_system_instruction(from_lang)},
                            {"role": "user", "content": [
                                {"type": "text", "text": refine_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                            ]}
                        ],
                        temperature=self.refine_temperature,
                        max_completion_tokens=self.max_tokens,
                        response_format=self.REFINE_RESPONSE_SCHEMA,
                    )
                    
                    if response and response.choices and response.choices[0].message.content:
                        raw_content = response.choices[0].message.content

                        # 检测拒绝回应的逻辑已移至 _parse_json_response
                        # The logic for detecting refusal messages has been moved to _parse_json_response

                        # Parse and obtain reordered texts & position mapping (single tolerant parser)
                        reordered_texts, original_position_mapping = self._parse_json_response(raw_content, queries)
                        self.logger.info(f"Stage 1 completed successfully: {len(reordered_texts)} texts reordered")
                        self.logger.debug(f"Reordered texts: {reordered_texts}")
                        self.logger.debug(f"Position mapping: {original_position_mapping}")
                        break # Success, exit retry loop
                    else:
                        raise ValueError("No content from vision model response")
                    
                except RefusalMessageError as e:
                    self.logger.warning(f"Stage 1 model refusal detected: {e}. Attempting fallback model (if configured).")
                    reordered_texts, original_position_mapping = await self._attempt_fallback_stage1(
                        refine_prompt, base64_img, from_lang, queries)
                    break # 不再重试主模型

                # 其它异常：先重试，最终再尝试 fallback
                except Exception as e:
                    if retry_count < self.stage1_retry_count:
                        self.logger.warning(
                            f"Stage 1 refinement failed (attempt {retry_count + 1}/{self.stage1_retry_count + 1}): {e}. Retrying...")
                        await asyncio.sleep(2 ** retry_count)  # 指数退避
                        continue  # 继续下一次循环
                    else:
                        self.logger.warning(
                            f"Stage 1 refinement failed (attempt {retry_count + 1}/{self.stage1_retry_count + 1}): {e}. All attempts failed.")
                        reordered_texts, original_position_mapping = await self._attempt_fallback_stage1(
                            refine_prompt, base64_img, from_lang, queries)
                        break  # 结束 retry 循环
            
            # Process refined output (remove unpaired symbols, etc.)
            reordered_texts = self._process_refine_output(reordered_texts)

            # Generate bboxes_fixed.png showing corrected text region order (only in verbose mode)
            if hasattr(ctx, 'result_path_callback') and hasattr(ctx, 'img_rgb') and query_regions and hasattr(ctx, 'verbose') and ctx.verbose:
                try:
                    import cv2
                    from ..utils.textblock import visualize_textblocks

                    # Create reordered text regions based on reading order
                    reordered_regions = []
                    for reading_idx, original_idx in enumerate(original_position_mapping):
                        if original_idx < len(query_regions) and query_regions[original_idx] is not None:
                            region = query_regions[original_idx]
                            # Update text with corrected version
                            if reading_idx < len(reordered_texts):
                                region.text = reordered_texts[reading_idx]
                            reordered_regions.append(region)

                    if reordered_regions:
                        # Generate visualization with corrected order (same numbering as reordered)
                        canvas = cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB).copy()
                        bboxes_fixed = visualize_textblocks(canvas, reordered_regions)

                        # Save using parent's result path callback
                        result_path = ctx.result_path_callback('bboxes_fixed.png')
                        cv2.imwrite(result_path, bboxes_fixed)
                except Exception as e:
                    self.logger.debug(f"Failed to generate bboxes_fixed.png: {e}")

            # Stage 2: Translation using reordered text
            self.logger.info(f"Stage 2: Translating reordered text using {self.stage2_model}...")
            
            # 术语表将通过系统消息自动应用，无需预处理文本
            # Glossary will be applied automatically through system messages, no need to preprocess text
            
            # 设置第二阶段翻译标志位和图片数据 / Set stage 2 translation flags and image data
            self._is_stage2_translation = True
            self._stage2_image_base64 = base64_img

            try:
                # Use parent class translation logic with reordered texts
                reordered_translations = await super()._translate(from_lang, to_lang, reordered_texts)
            except Exception as e:
                # Stage 2 翻译失败，清除标志位后重试，避免分割翻译时发送图片
                self.logger.warning(f"Stage 2 translation failed: {e}. Clearing stage 2 flags and retrying with text-only split translation.")
                self._is_stage2_translation = False
                self._stage2_image_base64 = None
                self._stage2_use_fallback = False

                try:
                    # 重新尝试翻译，此时不会发送图片
                    reordered_translations = await super()._translate(from_lang, to_lang, reordered_texts)
                except Exception as retry_e:
                    # 如果重试也失败，恢复标志位并重新抛出异常
                    self._is_stage2_translation = True
                    self._stage2_image_base64 = base64_img
                    raise retry_e
            finally:
                # 清除第二阶段翻译标志位和图片数据 / Clear stage 2 translation flags and image data
                self._is_stage2_translation = False
                self._stage2_image_base64 = None
                self._stage2_use_fallback = False # 重置回退状态
            
            # Remap translations back to original positions
            self.logger.info("Stage 3: Remapping translations to original positions...")
            final_translations = self._remap_translations_to_original_positions(
                reordered_translations, original_position_mapping
            )
            
            self.logger.info(f"2-stage translation completed: {len(queries)} texts processed with position mapping")
            self.logger.debug(f"Final translations in original order: {len(final_translations)} results")
            return final_translations
            
        except Exception as e:
            self.logger.error(f"2-stage translation failed: {e}. Falling back to single-stage.")
            return await super()._translate(from_lang, to_lang, queries)

    def _process_refine_output(self, refine_output: List[str]) -> List[str]:
        """
        Process refined output to remove unpaired symbols and clean text
        """
        all_symbols = self._LEFT_SYMBOLS + self._RIGHT_SYMBOLS
        processed = []

        for text in refine_output:
            stripped = text.strip()
            if removed := text[:len(text) - len(stripped)]:
                self.logger.debug(f'Removed leading characters: "{removed}" from "{text}"')

            left_count = sum(stripped.count(s) for s in self._LEFT_SYMBOLS)
            right_count = sum(stripped.count(s) for s in self._RIGHT_SYMBOLS)

            if left_count != right_count:
                for s in all_symbols:
                    stripped = stripped.replace(s, '')
                self.logger.debug(f'Removed unpaired symbols from "{stripped}"')

            processed.append(stripped.strip())
        return processed

    def _get_refine_prompt(self, text_regions, width: int, height: int, new_width: int, new_height: int):
        """
        Generate prompt for the refinement stage
        """
        lines = ["["]
        for i, region in enumerate(text_regions):
            if region is None:
                # Handle case where no matching region was found
                lines.append(f'\t{{"bbox_id": {i}, "bbox_2d": [0, 0, 100, 100], "text": ""}},')
            else:
                x1, y1, x2, y2 = region.xyxy
                x1, y1 = int((x1 / width) * new_width), int((y1 / height) * new_height)
                x2, y2 = int((x2 / width) * new_width), int((y2 / height) * new_height)
                lines.append(f'\t{{"bbox_id": {i}, "bbox_2d": [{x1}, {y1}, {x2}, {y2}], "text": "{region.text}"}},')
        
        # Remove trailing comma from last item
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
            
        lines.append("]")
        return "\n".join(lines)

    def _get_refine_system_instruction(self, from_lang: str):
        """
        System instruction for the OCR correction and text region reordering stage
        """
 
        return f"""你是专业的漫画文本处理引擎，负责OCR和文本区域排序纠正。    

**主要任务：**
1. **OCR错误纠正** - 修正字符识别错误、分割错误等
2. **文本区域重新排序** - 按照正确的阅读顺序重新排列

**排序示例：**
如果原始顺序是[0,1,2]，但正确阅读顺序应该是[2,0,1]，则：
- reading_order=2对应original_bbox_id=0
- reading_order=0对应original_bbox_id=1  
- reading_order=1对应original_bbox_id=2

**关键要求：**
1. reading_order从0开始，按正确阅读顺序递增，排序需注意分镜和气泡框的类型，相似气泡框为相连内容，同一分镜为一个整体
2. original_bbox_id保持原始编号
3. 排序时考虑气泡框的类型
4. 返回纯JSON格式，无其他内容
**重要：确保所有文本区域都有对应的条目，强制要求JSON格式输出。**

    **输出格式：**
    {{
    "corrected_regions": [
        {{
        "reading_order": 0,  # 阅读ID
        "original_bbox_id": 0, # 原始ID
        "bbox_2d": [x1, y1, x2, y2],
        "text": "纠正前的文本",
        "corrected_text": "纠正后的文本"
        }},
        {{
        "reading_order": 1,  # 阅读ID
        "original_bbox_id": 1, # 原始ID
        "bbox_2d": [x1, y1, x2, y2],
        "text": "另一个纠正前的文本",
        "corrected_text": "另一个纠正后的文本"
        }},
        ...
    ],
    "image_received": boolean # 是否接收到了图片数据
    }}   
"""



    # NOTE: strict structured parser removed; tolerant parser `_parse_json_response` is now the sole handler
    def _parse_json_response(self, raw_content: str, fallback_queries: List[str]) -> tuple[List[str], List[int]]:
        """
        Parse JSON response from vision model, handling new format with reading order and position mapping
        Returns: (reordered_texts, original_position_mapping)
        """
        try:
            # Step 1: 在解析前先检查是否为拒绝消息
            if self._contains_refusal(raw_content):
                raise RefusalMessageError(f"Refusal message detected: '{raw_content}'")

            # Step 2: Remove markdown code blocks and clean up
            cleaned = raw_content.strip()
            
            # Remove ```json and ``` markers
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            
            # Fix common JSON format errors
            cleaned = re.sub(r'"corr\{', '"corrected_regions": [', cleaned) 
       
            # Remove any text before the first [ or {
            match = re.search(r'(\[|\{)', cleaned)
            if match:
                cleaned = cleaned[match.start():]
            
            # Remove any text after the last ] or }
            # Find the last closing bracket/brace
            last_bracket = max(cleaned.rfind(']'), cleaned.rfind('}'))
            if last_bracket != -1:
                cleaned = cleaned[:last_bracket + 1]
            
            cleaned = cleaned.strip()
            
            self.logger.debug(f"Parsed JSON after cleanup: {cleaned}")
            
            # Step 3: Parse JSON
            data = json.loads(cleaned)
            
            # Step 4: Extract corrected texts - ignore key names, just find the array
            corrected_regions = []
            regions_array = None
            
            if isinstance(data, dict):
                # Find any array value that looks like regions data
                for value in data.values():
                    if isinstance(value, list) and value:
                        first_item = value[0]
                        if (isinstance(first_item, dict) and 
                            'reading_order' in first_item and 
                            'original_bbox_id' in first_item and 
                            'corrected_text' in first_item):
                            regions_array = value
                            break
                
                if regions_array:
                    for item in regions_array:
                        if isinstance(item, dict):
                            reading_order = item.get('reading_order', -1)
                            original_bbox_id = item.get('original_bbox_id', -1)
                            corrected_text = item.get('corrected_text', '').replace('\n', ' ').strip()
                            corrected_regions.append((reading_order, original_bbox_id, corrected_text))
                        
            elif isinstance(data, list):
                # Fallback: try to parse as old format array
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        bbox_id = item.get('bbox_id', i)
                        corrected = item.get('corrected_text', item.get('text', ''))
                        corrected_regions.append((i, bbox_id, corrected.replace('\n', ' ').strip()))
                    else:
                        corrected_regions.append((i, i, str(item)))
                        
            elif isinstance(data, dict):
                # Other object formats (fallback)
                if 'bboxes' in data:
                    for i, item in enumerate(data['bboxes']):
                        bbox_id = item.get('bbox_id', i)
                        corrected = item.get('corrected_text', item.get('text', ''))
                        corrected_regions.append((i, bbox_id, corrected.replace('\n', ' ').strip()))
                else:
                    # Single object format
                    bbox_id = data.get('bbox_id', 0)
                    corrected = data.get('corrected_text', data.get('text', ''))
                    corrected_regions.append((0, bbox_id, corrected.replace('\n', ' ').strip()))
            
            # Step 5: Sort by reading_order to get proper reading sequence
            corrected_regions.sort(key=lambda x: x[0] if x[0] >= 0 else 999)
            
            # Step 6: Extract reordered texts and position mapping
            reordered_texts = []
            original_position_mapping = []  # [reading_order_index] -> original_bbox_id
            
            for reading_order, original_bbox_id, corrected_text in corrected_regions:
                reordered_texts.append(corrected_text)
                original_position_mapping.append(original_bbox_id)
            
            # Step 7: Validate and handle edge cases
            expected_count = len(fallback_queries)
            
            if len(reordered_texts) != expected_count:
                self.logger.warning(f"Expected {expected_count} texts but got {len(reordered_texts)}, falling back to original")
                return fallback_queries, list(range(expected_count))
            
            # Validate that all original bbox IDs are present
            expected_bbox_ids = set(range(expected_count))
            actual_bbox_ids = set(original_position_mapping)
            
            if expected_bbox_ids != actual_bbox_ids:
                self.logger.warning(f"Missing or invalid bbox IDs. Expected: {expected_bbox_ids}, Got: {actual_bbox_ids}")
                return fallback_queries, list(range(expected_count))
            
            self.logger.info(f"Successfully parsed {len(reordered_texts)} texts with position mapping: {original_position_mapping}")
            return reordered_texts, original_position_mapping
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.debug(f"Failed to parse: {raw_content}")
            # 抛出异常以触发重试，而不是回退到fallback
            raise e
            
        except RefusalMessageError: # 重新抛出以确保被上层捕获
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error in JSON parsing: {e}")
            # 抛出异常以触发重试
            raise e

    def _remap_translations_to_original_positions(self, reordered_translations: List[str], 
                                                original_position_mapping: List[int]) -> List[str]:
        """
        Remap translations from reading order back to original positions
        
        Args:
            reordered_translations: Translations in reading order
            original_position_mapping: [reading_order_index] -> original_bbox_id
            
        Returns:
            Translations in original position order
        """
        try:
            # Create a mapping from original_bbox_id to translation
            bbox_to_translation = {}
            for reading_idx, original_bbox_id in enumerate(original_position_mapping):
                if reading_idx < len(reordered_translations):
                    bbox_to_translation[original_bbox_id] = reordered_translations[reading_idx]
            
            # Rebuild translations in original order (0, 1, 2, ...)
            final_translations = []
            for original_idx in range(len(original_position_mapping)):
                if original_idx in bbox_to_translation:
                    final_translations.append(bbox_to_translation[original_idx])
                else:
                    # Fallback: use empty string or original if available
                    final_translations.append("")
                    self.logger.warning(f"No translation found for original position {original_idx}")
            
            self.logger.info(f"Remapped {len(reordered_translations)} translations to original positions")
            self.logger.debug(f"Position mapping: {original_position_mapping}")
            self.logger.debug(f"Final translations order: {[t[:20] + '...' if len(t) > 20 else t for t in final_translations]}")
            
            return final_translations
            
        except Exception as e:
            self.logger.error(f"Error in position remapping: {e}")
            # Fallback: return translations as-is
            return reordered_translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        """
        重写父类的_request_translation方法，在第二阶段翻译时发送图片
        """
        lang_name = self._LANGUAGE_CODE_MAP.get(to_lang, to_lang) if to_lang in self._LANGUAGE_CODE_MAP else to_lang
                
        # 构建 messages / Construct messages
        messages = [  
            {'role': 'system', 'content': self.chat_system_template.format(to_lang=lang_name)},  
        ]  

        # 提取相关术语并添加到系统消息中  / Extract relevant terms and add them to the system message
        has_glossary = False  # 添加标志表示是否有术语表 / Add a flag to indicate whether there is a glossary
        relevant_terms = self.extract_relevant_terms(prompt)  
        if relevant_terms:  
            has_glossary = True  # 设置标志 / Set the flag
            # 构建术语表字符串 / Construct the glossary string
            glossary_text = "\n".join([f"{term}->{translation}" for term, translation in relevant_terms.items()])  
            system_message = self.glossary_system_template.format(glossary_text=glossary_text)  
            messages.append({'role': 'system', 'content': system_message})  
            self.logger.info(f"Loaded {len(relevant_terms)} relevant terms from the glossary.")  
        
        # 如果有上文，添加到系统消息中 / If there is a previous context, add it to the system message        
        if self.prev_context:
            messages.append({'role': 'system', 'content': self.prev_context})            
        
        # 如果需要先给出示例对话
        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)

        # 如果需要先给出示例对话 / Provide an example dialogue first if necessary
        if hasattr(self, 'chat_sample') and lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        # 构建用户消息 - 第二阶段时根据配置决定是否包含图片 / Construct user message - include image in stage 2 based on config
        if self._is_stage2_translation and self.stage2_send_image and not self._stage2_use_fallback:
            # Check if this is batch processing
            if hasattr(self, '_stage2_batch_images') and self._stage2_batch_images:
                # Batch Stage 2: Send text and multiple images
                user_content = [{'type': 'text', 'text': prompt}]
                for base64_img in self._stage2_batch_images:
                    user_content.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{base64_img}'}
                    })
                user_message = {'role': 'user', 'content': user_content}
                messages.append(user_message)
            elif self._stage2_image_base64:
                # Single image Stage 2: Send text and single image
                user_message = {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{self._stage2_image_base64}'}}
                    ]
                }
                messages.append(user_message)
            else:
                # No image available, send text only
                messages.append({'role': 'user', 'content': prompt})
        else:
            # 普通翻译或禁用图片：只发送文本 / Normal translation or image disabled: send text only
            messages.append({'role': 'user', 'content': prompt})

        # 准备输出的 prompt 文本 / Prepare the output prompt text 
        if self.verbose_logging:  
            # 在详细模式下，也要处理包含图片的消息，避免显示巨大的base64数据
            verbose_msgs = []
            for m in messages:
                content = m['content']
                if isinstance(content, list):
                    # 处理包含图片的消息 - 只显示文本部分
                    text_content = next((item['text'] for item in content if item['type'] == 'text'), '')
                    verbose_msgs.append(f"{m['role'].upper()}:\n{text_content}")
                else:
                    verbose_msgs.append(f"{m['role'].upper()}:\n{content}")
            
            prompt_text = "\n".join(verbose_msgs)
            
            # 在第二阶段添加图片发送提醒（verbose模式）
            if self._is_stage2_translation and self.stage2_send_image and not self._stage2_use_fallback:
                if hasattr(self, '_stage2_batch_images') and self._stage2_batch_images:
                    prompt_text += f"\n[IMAGES: {len(self._stage2_batch_images)} manga pages sent with batch translation request]"
                elif self._stage2_image_base64:
                    prompt_text += "\n[IMAGE: Original manga page sent with translation request]"
            elif self._is_stage2_translation and (not self.stage2_send_image or self._stage2_use_fallback):
                if self._stage2_use_fallback:
                    prompt_text += "\n[IMAGE: Disabled for fallback model - Text-only translation request]"
                else:
                    prompt_text += "\n[IMAGE: Disabled - Text-only translation request]"
            
            self.print_boxed(prompt_text, border_color="cyan", title="GPT Prompt")      
        else:  
            simplified_msgs = []  
            for i, m in enumerate(messages):  
                if (has_glossary and i == 1) or (i == len(messages) - 1):  
                    content = m['content']
                    if isinstance(content, list):
                        # 处理包含图片的消息
                        text_content = next((item['text'] for item in content if item['type'] == 'text'), '')
                        simplified_msgs.append(f"{m['role'].upper()}:\n{text_content}")
                    else:
                        simplified_msgs.append(f"{m['role'].upper()}:\n{content}")
                else:  
                    simplified_msgs.append(f"{m['role'].upper()}:\n[HIDDEN CONTENT]")
            
            prompt_text = "\n".join(simplified_msgs)
            
            # 在第二阶段添加图片发送提醒
            if self._is_stage2_translation and self.stage2_send_image and not self._stage2_use_fallback:
                if hasattr(self, '_stage2_batch_images') and self._stage2_batch_images:
                    prompt_text += f"\n[IMAGES: {len(self._stage2_batch_images)} manga pages sent with batch translation request]"
                elif self._stage2_image_base64:
                    prompt_text += "\n[IMAGE: Original manga page sent with translation request]"
            elif self._is_stage2_translation and (not self.stage2_send_image or self._stage2_use_fallback):
                if self._stage2_use_fallback:
                    prompt_text += "\n[IMAGE: Disabled for fallback model - Text-only translation request]"
                else:
                    prompt_text += "\n[IMAGE: Disabled - Text-only translation request]"
            
            # 使用 rich 输出 prompt / Use rich to output the prompt
            self.print_boxed(prompt_text, border_color="cyan", title="GPT Prompt (verbose=False)") 
        

        # 发起请求 / Initiate the request
        # 在Stage 2时使用指定的Stage 2模型或已激活的fallback模型
        model_to_use = OPENAI_MODEL
        if self._is_stage2_translation:
            if self._stage2_use_fallback and hasattr(self, '_fallback_model') and self._fallback_model:
                model_to_use = self._fallback_model
                self.logger.info(f"Using activated fallback model for Stage 2 (text-only mode): {model_to_use}")
            else:
                model_to_use = self.stage2_model
        else:
            # For non-stage2, use the default model from parent logic, which is typically OPENAI_MODEL
            # This branch is needed to avoid using a potentially uninitialized model_to_use
            model_to_use = OPENAI_MODEL

        response = await self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self._TIMEOUT
        )

        if not response.choices:
            raise ValueError("Empty response from OpenAI API")

        raw_text = response.choices[0].message.content

        # 新增：检测Stage 2的拒绝回应，并激活fallback
        if self._is_stage2_translation and not self._stage2_use_fallback:
            has_numeric_prefix = re.search(r'<\|(\d+)\|>', raw_text)
            if not has_numeric_prefix and self._contains_refusal(raw_text):
                if hasattr(self, '_fallback_model') and self._fallback_model:
                    self.logger.warning("Stage 2 refusal detected. Activating fallback model for subsequent requests.")
                    self._stage2_use_fallback = True
                    raise RefusalMessageError("Stage 2 refusal, switching to fallback model.")
                else:
                    self.logger.warning("Stage 2 refusal detected, but no fallback model is configured.")

        # 去除 <think>...</think> 标签及内容。由于某些中转api的模型的思考过程是被强制输出的，并不包含在reasoning_content中，需要额外过滤
        # Remove <think>...</think> tags and their contents. Since the reasoning process of some relay API models is forcibly output and not included in the reasoning_content, additional filtering is required.
        raw_text = re.sub(r'(</think>)?<think>.*?</think>', '', raw_text, flags=re.DOTALL)

        # 删除多余的空行 / Remove extra blank lines
        
        cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text).strip()

        # 删除数字前缀前后的不相关的解释性文字。但不出现数字前缀时，保留限制词防止删得什么都不剩
        # Remove irrelevant explanatory text before and after numerical prefixes. However, when numerical prefixes are not present, retain restrictive words to prevent deleting everything.
        lines = cleaned_text.splitlines()
        min_index_line_index = -1
        max_index_line_index = -1
        has_numeric_prefix = False  # Flag to check if any numeric prefix exists

        for index, line in enumerate(lines):
            match = re.search(r'<\|(\d+)\|>', line)
            if match:
                has_numeric_prefix = True
                current_index = int(match.group(1))
                if current_index == 1:  # 查找最小标号 <|1|> / find <|1|>
                    min_index_line_index = index
                if max_index_line_index == -1 or current_index > int(re.search(r'<\|(\d+)\|>', lines[max_index_line_index]).group(1)):  # 查找最大标号 / find max number
                    max_index_line_index = index
                    
        if has_numeric_prefix:
            modified_lines = []
            if min_index_line_index != -1:
                modified_lines.extend(lines[min_index_line_index:])  # 从最小标号行开始保留到结尾 / Keep from the row with the smallest label to the end

            if max_index_line_index != -1 and modified_lines:  # 确保 modified_lines 不为空，且找到了最大标号 / Ensure that modified_lines is not empty and that the maximum label has been found
                modified_lines = modified_lines[:max_index_line_index - min_index_line_index + 1]  # 只保留到最大标号行 (相对于 modified_lines 的索引) / Retain only up to the row with the maximum label (relative to the index of modified_lines)

            cleaned_text = "\n".join(modified_lines)      
        
        # 记录 token 消耗 / Record token consumption
        if not hasattr(response, 'usage') or not hasattr(response.usage, 'total_tokens'):
            self.logger.warning("Response does not contain usage information") #第三方逆向中转api不返回token数 / The third-party reverse proxy API does not return token counts
            self.token_count_last = 0
        else:
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens
        
        response_text = cleaned_text
        self.print_boxed(response_text, border_color="green", title="GPT Response")          
        return cleaned_text

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], ctx: Context, use_mtpe: bool = False) -> List[str]:
        """
        Main translation entry point - override to ensure context is passed through
        """
        self._stage2_use_fallback = False # 确保每次外部调用都重置状态
        if not queries:
            return queries

        # Auto-detect language if needed
        if from_lang == 'auto':
            from_langs = []
            for region in ctx.text_regions if ctx and ctx.text_regions else []:
                for lang, pattern in self._LANG_PATTERNS:
                    if re.search(pattern, region.text):
                        from_langs.append(lang)
                        break
                else:
                    from_langs.append('ENG')
            from_lang = Counter(from_langs).most_common(1)[0][0] if from_langs else 'ENG'

        from_lang_name = self._LANGUAGE_CODE_MAP.get(from_lang, from_lang)
        to_lang_name = self._LANGUAGE_CODE_MAP.get(to_lang, to_lang)
        
        if from_lang_name == to_lang_name:
            return queries

        # Filter out non-valuable text
        query_indices, final_translations = [], []
        for i, q in enumerate(queries):
            final_translations.append(queries[i] if not is_valuable_text(q) else None)
            if is_valuable_text(q):
                query_indices.append(i)

        filtered_queries = [queries[i] for i in query_indices]
        
        if not filtered_queries:
            return final_translations

        # Perform 2-stage translation
        await self._ratelimit_sleep()
        translations = await self._translate(from_lang, to_lang, filtered_queries, ctx)

        # Apply post-processing
        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(filtered_queries, translations)]

        # Handle Arabic reshaping if needed
        if to_lang == 'ARA':
            try:
                import arabic_reshaper
                translations = [arabic_reshaper.reshape(t) for t in translations]
            except ImportError:
                self.logger.warning("arabic_reshaper not available for Arabic text reshaping")

        # Apply MTPE if requested
        if use_mtpe and hasattr(self, 'mtpe_adapter'):
            translations = await self.mtpe_adapter.dispatch(filtered_queries, translations)

        # Reconstruct final results
        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f'{i}: {filtered_queries[i]} => {trans}')

        return final_translations

    async def _translate_batch_2stage(self, from_lang: str, to_lang: str, queries: List[str], batch_contexts: List[Context]) -> List[str]:
        """
        Batch processing version of 2-stage translation:
        1. Stage 1: OCR correction and text region reordering for multiple images
        2. Stage 2: Translation using reordered text with multiple images
        3. Stage 3: Remap translations back to original positions
        """
        try:
            self.logger.info(f"Starting batch 2-stage translation for {len(batch_contexts)} images with {len(queries)} total queries")

            # Collect all images and text regions
            batch_images = []
            batch_query_regions = []
            query_to_image_mapping = []  # Maps query index to (image_index, region_index)

            query_idx = 0
            for img_idx, ctx in enumerate(batch_contexts):
                rgb_img = Image.fromarray(ctx.img_rgb)
                batch_images.append(rgb_img)

                # Get text regions for this image
                num_queries_for_image = len([q for q in queries[query_idx:] if query_idx < len(queries)])
                if ctx.text_regions:
                    image_regions = ctx.text_regions[:num_queries_for_image] if query_idx < len(queries) else []
                else:
                    image_regions = []

                # Calculate how many queries belong to this image
                queries_for_this_image = min(len(image_regions), len(queries) - query_idx)

                for region_idx in range(queries_for_this_image):
                    if query_idx < len(queries):
                        query_to_image_mapping.append((img_idx, region_idx))
                        query_idx += 1

                batch_query_regions.append(image_regions[:queries_for_this_image])

            # Stage 1: Batch OCR correction and text reordering
            self.logger.info(f"Stage 1: Batch OCR correction for {len(batch_images)} images using {self.stage1_model}...")

            # Encode all images
            batch_base64_images = []
            batch_dimensions = []
            for rgb_img in batch_images:
                w, h = rgb_img.size
                base64_img, nw, nh = encode_image(rgb_img)
                batch_base64_images.append(base64_img)
                batch_dimensions.append((w, h, nw, nh))

            # Create batch refine prompt
            batch_refine_prompt = self._get_batch_refine_prompt(batch_query_regions, batch_dimensions)

            self.logger.info("Stage 1 Batch OCR Request - JSON Content:")
            self.logger.info(f"{batch_refine_prompt}")

            # Default fallback values
            batch_reordered_texts = [queries[i] if i < len(queries) else "" for i in range(len(query_to_image_mapping))]
            batch_position_mapping = list(range(len(query_to_image_mapping)))

            # Try batch Stage 1 processing
            response = None
            for retry_count in range(self.stage1_retry_count + 1):
                try:
                    await self._ratelimit_sleep()

                    # Construct messages with multiple images
                    user_content = [{"type": "text", "text": batch_refine_prompt}]
                    for base64_img in batch_base64_images:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        })

                    response = await self.client.chat.completions.create(
                        model=self.stage1_model,
                        messages=[
                            {"role": "system", "content": self._get_batch_refine_system_instruction(from_lang)},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=self.refine_temperature,
                        max_completion_tokens=self.max_tokens,
                        response_format=self.BATCH_REFINE_RESPONSE_SCHEMA,
                    )

                    if response and response.choices and response.choices[0].message.content:
                        raw_content = response.choices[0].message.content

                        # Check for refusal messages
                        if self._contains_refusal(raw_content):
                            raise RefusalMessageError(f"Batch Stage 1 refusal message detected: '{raw_content}'")

                        # Log the raw response content for debugging (similar to single image processing)
                        self.logger.info("Parsed JSON after cleanup:")
                        try:
                            # Clean up and parse JSON for display
                            cleaned = raw_content.strip()
                            cleaned = re.sub(r'```json\s*', '', cleaned)
                            cleaned = re.sub(r'```\s*$', '', cleaned)
                            import json
                            parsed_data = json.loads(cleaned)
                            self.logger.info(json.dumps(parsed_data, indent=2, ensure_ascii=False))
                        except Exception as e:
                            self.logger.debug(f"Failed to parse JSON for display: {e}")
                            self.logger.info(raw_content)

                        batch_reordered_texts, batch_position_mapping = self._parse_batch_json_response(
                            raw_content, queries, query_to_image_mapping
                        )
                        self.logger.info(f"Successfully parsed {len(batch_reordered_texts)} texts with position mapping: {batch_position_mapping}")
                        self.logger.info(f"Batch Stage 1 completed successfully: {len(batch_reordered_texts)} texts reordered")
                        self.logger.debug(f"Reordered texts: {batch_reordered_texts}")
                        self.logger.debug(f"Position mapping: {batch_position_mapping}")
                        break
                    else:
                        raise ValueError("No content from vision model response")

                except RefusalMessageError as e:
                    self.logger.warning(f"Batch Stage 1 model refusal detected: {e}. Attempting batch fallback model (if configured).")
                    # Try batch fallback model
                    batch_reordered_texts, batch_position_mapping = await self._attempt_batch_fallback_stage1(
                        batch_refine_prompt, batch_base64_images, from_lang, queries, query_to_image_mapping)
                    break

                except Exception as e:
                    if retry_count < self.stage1_retry_count:
                        self.logger.warning(f"Batch Stage 1 failed (attempt {retry_count + 1}/{self.stage1_retry_count + 1}): {e}. Retrying...")
                        await asyncio.sleep(2 ** retry_count)
                        continue
                    else:
                        self.logger.warning(f"Batch Stage 1 failed after all attempts: {e}. Attempting batch fallback model (if configured).")

                        # Try batch fallback model
                        batch_reordered_texts, batch_position_mapping = await self._attempt_batch_fallback_stage1(
                            batch_refine_prompt, batch_base64_images, from_lang, queries, query_to_image_mapping)
                        break

            # Process refined output
            batch_reordered_texts = self._process_refine_output(batch_reordered_texts)

            # Stage 2: Batch translation using reordered text
            self.logger.info(f"Stage 2: Batch translating reordered text using {self.stage2_model}...")

            # Set batch stage 2 translation flags
            self._is_stage2_translation = True
            self._stage2_batch_images = batch_base64_images

            try:
                # Use parent class translation logic with reordered texts
                batch_reordered_translations = await super()._translate(from_lang, to_lang, batch_reordered_texts)
            except Exception as e:
                # Batch Stage 2 翻译失败，清除标志位后重试，避免分割翻译时发送图片
                self.logger.warning(f"Batch Stage 2 translation failed: {e}. Clearing stage 2 flags and retrying with text-only split translation.")
                self._is_stage2_translation = False
                self._stage2_batch_images = None
                self._stage2_use_fallback = False

                try:
                    # 重新尝试翻译，此时不会发送图片
                    batch_reordered_translations = await super()._translate(from_lang, to_lang, batch_reordered_texts)
                except Exception as retry_e:
                    # 如果重试也失败，恢复标志位并重新抛出异常
                    self._is_stage2_translation = True
                    self._stage2_batch_images = batch_base64_images
                    raise retry_e
            finally:
                # Clear batch stage 2 translation flags
                self._is_stage2_translation = False
                self._stage2_batch_images = None

            # Stage 3: Remap translations back to original positions and generate debug images
            self.logger.info("Stage 3: Remapping batch translations to original positions...")
            final_translations = [''] * len(queries)

            for i, original_pos in enumerate(batch_position_mapping):
                if original_pos < len(queries) and i < len(batch_reordered_translations):
                    final_translations[original_pos] = batch_reordered_translations[i]

            # Generate bboxes_fixed.png for each image in batch
            await self._generate_batch_debug_images(batch_contexts, batch_query_regions, batch_reordered_texts, batch_position_mapping, query_to_image_mapping)

            self.logger.info(f"Batch remapped {len(final_translations)} translations to original positions")
            return final_translations

        except Exception as e:
            self.logger.error(f"Batch 2-stage translation failed: {e}")
            # Fallback to individual processing
            self.logger.info("Falling back to individual image processing...")
            results = []
            query_idx = 0
            for ctx in batch_contexts:
                ctx_queries = []
                if ctx.text_regions:
                    for _ in ctx.text_regions:
                        if query_idx < len(queries):
                            ctx_queries.append(queries[query_idx])
                            query_idx += 1

                if ctx_queries:
                    ctx_results = await self._translate_2stage(from_lang, to_lang, ctx_queries, ctx)
                    results.extend(ctx_results)

            return results

    def _get_batch_refine_prompt(self, batch_query_regions: List[List], batch_dimensions: List[tuple]):
        """
        Generate prompt for batch refinement stage with multiple images
        """
        lines = ["["]
        bbox_id = 0

        for img_idx, (query_regions, (width, height, new_width, new_height)) in enumerate(zip(batch_query_regions, batch_dimensions)):
            for region_idx, region in enumerate(query_regions):
                if region is None:
                    lines.append(f'\t{{"bbox_id": {bbox_id}, "image_index": {img_idx}, "bbox_2d": [0, 0, 100, 100], "text": ""}},')
                else:
                    x1, y1, x2, y2 = region.xyxy
                    x1, y1 = int((x1 / width) * new_width), int((y1 / height) * new_height)
                    x2, y2 = int((x2 / width) * new_width), int((y2 / height) * new_height)
                    lines.append(f'\t{{"bbox_id": {bbox_id}, "image_index": {img_idx}, "bbox_2d": [{x1}, {y1}, {x2}, {y2}], "text": "{region.text}"}},')
                bbox_id += 1

        # Remove trailing comma from last item
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]

        lines.append("]")
        return "\n".join(lines)

    def _get_batch_refine_system_instruction(self, from_lang: str):
        """
        System instruction for batch OCR correction and text region reordering stage
        """
        return f"""你是专业的漫画文本处理引擎，负责批量处理多张图片的OCR和文本区域排序纠正。

**主要任务：**
1. **批量OCR错误纠正** - 修正多张图片中的字符识别错误、分割错误等
2. **批量文本区域重新排序** - 按照每张图片内正确的阅读顺序重新排列

**输入格式：**
- 你将收到多张图片和对应的文本区域JSON数据
- 每个文本区域包含：bbox_id（全局唯一ID）、image_index（图片索引）、bbox_2d（坐标）、text（OCR文本）

**排序示例：**
如果图片0的原始顺序是[0,1,2]，但正确阅读顺序应该是[2,0,1]，则：
- reading_order=0对应original_bbox_id=2
- reading_order=1对应original_bbox_id=0
- reading_order=2对应original_bbox_id=1

**关键要求：**
1. 每张图片内reading_order从0开始，按正确阅读顺序递增
2. original_bbox_id保持输入的bbox_id
3. 排序时考虑气泡框的类型和分镜结构
4. 返回纯JSON格式，无其他内容
5. 保留所有输入的文本区域，即使是空文本

**输出格式：**
{{
  "batch_results": [
    {{
      "image_index": 0,
      "corrected_regions": [
        {{
          "reading_order": 0,
          "original_bbox_id": 2,
          "bbox_2d": [x1, y1, x2, y2],
          "text": "图片0第一个要读的原文",
          "corrected_text": "图片0第一个要读的纠正文本"
        }},
        {{
          "reading_order": 1,
          "original_bbox_id": 0,
          "bbox_2d": [x1, y1, x2, y2],
          "text": "图片0第二个要读的原文",
          "corrected_text": "图片0第二个要读的纠正文本"
        }},
        {{
          "reading_order": 2,
          "original_bbox_id": 1,
          "bbox_2d": [x1, y1, x2, y2],
          "text": "图片0第三个要读的原文",
          "corrected_text": "图片0第三个要读的纠正文本"
        }}
      ]
    }},
    {{
      "image_index": 1,
      "corrected_regions": [
        {{
          "reading_order": 0,
          "original_bbox_id": 3,
          "bbox_2d": [x1, y1, x2, y2],
          "text": "图片1第一个要读的原文",
          "corrected_text": "图片1第一个要读的纠正文本"
        }},
        {{
          "reading_order": 1,
          "original_bbox_id": 4,
          "bbox_2d": [x1, y1, x2, y2],
          "text": "图片1第二个要读的原文",
          "corrected_text": "图片1第二个要读的纠正文本"
        }}
      ]
    }}
  ],
  "images_received": 2
}}

**语言：{from_lang}**
**重要：确保所有文本区域都有对应的条目，强制要求JSON格式输出。**"""

    def _parse_batch_json_response(self, raw_content: str, fallback_queries: List[str], query_to_image_mapping: List[tuple]) -> tuple[List[str], List[int]]:
        """
        Parse batch JSON response from vision model
        Returns: (reordered_texts, original_position_mapping)
        """
        try:
            # Check for refusal messages
            if self._contains_refusal(raw_content):
                raise RefusalMessageError(f"Refusal message detected: '{raw_content}'")

            # Clean up the response
            cleaned = raw_content.strip()
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)

            # Parse JSON
            data = json.loads(cleaned)

            if 'batch_results' not in data:
                raise ValueError("Missing 'batch_results' in response")

            # Initialize result arrays
            reordered_texts = []
            original_position_mapping = []

            # Process each image's results
            for image_result in data['batch_results']:
                image_index = image_result.get('image_index', 0)
                corrected_regions = image_result.get('corrected_regions', [])

                # Sort by reading order within this image
                corrected_regions.sort(key=lambda x: x.get('reading_order', 0))

                # Extract texts and create position mapping
                for region in corrected_regions:
                    original_bbox_id = region.get('original_bbox_id', 0)
                    corrected_text = region.get('corrected_text', region.get('text', ''))

                    reordered_texts.append(corrected_text)
                    original_position_mapping.append(original_bbox_id)

            self.logger.debug(f"Parsed batch JSON: {len(reordered_texts)} texts with position mapping: {original_position_mapping}")
            return reordered_texts, original_position_mapping

        except Exception as e:
            self.logger.warning(f"Failed to parse batch JSON response: {e}")
            self.logger.debug(f"Raw content: {raw_content}")

            # Fallback to original queries
            return fallback_queries, list(range(len(fallback_queries)))

    async def _generate_batch_debug_images(self, batch_contexts: List[Context], batch_query_regions: List[List],
                                         batch_reordered_texts: List[str], batch_position_mapping: List[int],
                                         query_to_image_mapping: List[tuple]):
        """
        Generate bboxes_fixed.png for each image in the batch (only in verbose mode)
        """
        try:
            import cv2
            from ..utils.textblock import visualize_textblocks

            # Group reordered texts and mappings by image
            image_results = {}
            for text_idx, (img_idx, region_idx) in enumerate(query_to_image_mapping):
                if img_idx not in image_results:
                    image_results[img_idx] = []

                if text_idx < len(batch_reordered_texts):
                    image_results[img_idx].append({
                        'region_idx': region_idx,
                        'reordered_text': batch_reordered_texts[text_idx],
                        'original_pos': batch_position_mapping[text_idx] if text_idx < len(batch_position_mapping) else text_idx
                    })

            # Generate debug image for each image (only in verbose mode)
            for img_idx, ctx in enumerate(batch_contexts):
                if img_idx not in image_results:
                    continue

                if not hasattr(ctx, 'result_path_callback') or not hasattr(ctx, 'img_rgb'):
                    continue

                # Check if verbose mode is enabled
                if not (hasattr(ctx, 'verbose') and ctx.verbose):
                    continue

                query_regions = batch_query_regions[img_idx] if img_idx < len(batch_query_regions) else []
                if not query_regions:
                    continue

                # Create reordered regions for this image
                reordered_regions = []
                image_data = image_results[img_idx]

                # Sort by original position to maintain correct order
                image_data.sort(key=lambda x: x['original_pos'])

                for data in image_data:
                    region_idx = data['region_idx']
                    if region_idx < len(query_regions) and query_regions[region_idx] is not None:
                        region = query_regions[region_idx]
                        # Create a copy and update text with corrected version
                        region_copy = region
                        region_copy.text = data['reordered_text']
                        reordered_regions.append(region_copy)

                if reordered_regions:
                    # Generate visualization with corrected order
                    canvas = cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB).copy()
                    bboxes_fixed = visualize_textblocks(canvas, reordered_regions)

                    # Save using the context's result path callback
                    result_path = ctx.result_path_callback('bboxes_fixed.png')
                    cv2.imwrite(result_path, bboxes_fixed)

        except Exception as e:
            self.logger.debug(f"Failed to generate batch debug images: {e}")