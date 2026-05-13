import re

from ..config import TranslatorConfig
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List
from .common import CommonTranslator, VALID_LANGUAGES
from .keys import CUSTOM_OPENAI_API_KEY, CUSTOM_OPENAI_API_BASE, CUSTOM_OPENAI_MODEL, CUSTOM_OPENAI_MODEL_CONF

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# Vietnamese has unique diacritical marks — their presence confirms Vietnamese output
_VI_DIACRITIC_RE = re.compile(
    r"[àáâãèéêìíòóôõùúăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]",
    re.IGNORECASE,
)
# 3+ letter Latin words — 2+ with no VI diacritics = likely English
_EN_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


def _strip_generation_artifacts(text: str, preserve_segment_tokens: bool = False) -> str:
    if not isinstance(text, str) or not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</\|\d+\|>?", "", cleaned)  # </|3|> and </|3|
    cleaned = re.sub(
        r"<\|(?:assistant|user|system|im_start|im_end|eot_id|end_of_text|endoftext|begin_of_text|bos|eos|pad|unk)\|>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"</?(?:s|bos|eos|pad|unk)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[/?INST\]|<<SYS>>|<</SYS>>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</(?=[^a-zA-Z]|$)", "", cleaned)
    if not preserve_segment_tokens:
        cleaned = re.sub(r"<\|\d+\|>?", "", cleaned)  # <|3|> and <|3|
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _contains_watermark_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    compact = re.sub(r"\s+", "", text.lower())
    compact = re.sub(r"[^a-z0-9._:/-]", "", compact)
    return (
        "acg" in compact
        or ".com" in compact
        or compact.endswith("com")
        or "www." in compact
        or "http://" in compact
        or "https://" in compact
    )


def _clean_watermark_fragments(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    lines = [line.strip() for line in _strip_generation_artifacts(text).splitlines() if line.strip()]
    filtered_lines = [line for line in lines if not _contains_watermark_text(line)]
    if filtered_lines:
        return "\n".join(filtered_lines)
    if _contains_watermark_text(text):
        return ""
    return _strip_generation_artifacts(text)


def _target_is_vietnamese(to_lang: str) -> bool:
    if not isinstance(to_lang, str):
        return False
    normalized = to_lang.strip().lower()
    return normalized in {"vin", "vi", "vietnamese", "tiếng việt", "tieng viet"}


def _needs_vietnamese_retry(text: str) -> bool:
    """True if text is likely English/Chinese rather than Vietnamese."""
    cleaned = _clean_watermark_fragments(_strip_generation_artifacts(text or ""))
    if not cleaned or len(cleaned.strip()) < 4:
        return False
    if _CHINESE_RE.search(cleaned):
        return True  # Untranslated Chinese
    if _VI_DIACRITIC_RE.search(cleaned):
        return False  # Vietnamese diacritics present = Vietnamese
    en_words = _EN_WORD_RE.findall(cleaned)
    return len(en_words) >= 2  # 2+ plain Latin words, no VI diacritics = English


class CustomOpenAiTranslator(ConfigGPT, CommonTranslator):
    _INVALID_REPEAT_COUNT = 2  # 如果检测到"无效"翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 40  # 每分钟最大请求次数
    _TIMEOUT = 240  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 15  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 32000

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    # Extra system content appended during Vietnamese retry — reset per prompt batch
    _vi_retry_extra_system: str = ''

    def __init__(self, model=None, api_base=None, api_key=None, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Use the `ollama` defaults.
        _CONFIG_KEY='ollama'
        if CUSTOM_OPENAI_MODEL_CONF:
            _CONFIG_KEY+=f".{CUSTOM_OPENAI_MODEL_CONF}"

        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        self.model = model
        CommonTranslator.__init__(self)
        self.client = openai.AsyncOpenAI(api_key=api_key or CUSTOM_OPENAI_API_KEY or "ollama") # required, but unused for ollama
        self.client.base_url = api_base or CUSTOM_OPENAI_API_BASE
        self.token_count = 0
        self.token_count_last = 0

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config

    def _get_vietnamese_retry_system_suffix(self) -> str:
        """Extra system message appended when retrying to enforce Vietnamese output.
        NOTE: This goes into the SYSTEM message, NOT the user prompt, to prevent
        the model from echoing these instructions as translation content.
        """
        return (
            "\n\nCRITICAL OVERRIDE — YOUR PREVIOUS RESPONSE WAS REJECTED:\n"
            "Your previous output contained non-Vietnamese text. This is unacceptable.\n"
            "Rules for this retry attempt:\n"
            "1. Translate ALL segments to natural Vietnamese.\n"
            "2. Keep ALL segment markers <|1|>, <|2|>, etc. in correct order.\n"
            "3. Watermark/logo segments (containing ACG, .com, .net, .org) → output empty string for that segment.\n"
            "4. NEVER output English, Chinese, or system tokens like </, </|3|>, <|assistant|>, </s>.\n"
            "5. Output ONLY the translated segments — do NOT echo or repeat these instructions."
        )

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        cleaned_query = _strip_generation_artifacts(query or "")
        cleaned_trans = _clean_watermark_fragments(_strip_generation_artifacts(trans or ""))

        # Invisible ZWJ watermark marker produced by our fallback → accept, no outer retry.
        if cleaned_trans == "\u200d":
            return False

        # Source-text fallback (translation identical to original) → accept, no outer retry.
        # MIT's post-processing will filter it as "Translation identical to original",
        # leaving the original Chinese text visible instead of rendering English.
        if cleaned_query.lower().strip() == cleaned_trans.lower().strip():
            return False

        if _contains_watermark_text(cleaned_query) and cleaned_trans in ("", "\u200d"):
            return False
        if _target_is_vietnamese(getattr(self, "_active_to_lang", "")) and _needs_vietnamese_retry(cleaned_trans):
            return True
        return super()._is_translation_invalid(cleaned_query, cleaned_trans)


    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ''

        if self._INCLUDE_TEMPLATE:
            prompt += self.prompt_template.format(to_lang=to_lang)

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i + 1 - i_offset}|>{query}'

            # If prompt is growing too large and there's still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i + 1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                yield prompt.lstrip(), i + 1 - i_offset
                prompt = self.prompt_template.format(to_lang=to_lang)
                # Restart counting at 1
                i_offset = i + 1

        if self._RETURN_PROMPT:
            prompt += '\n<|1|>'

        yield prompt.lstrip(), len(queries) - i_offset

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                prompt,
            ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self._active_to_lang = to_lang
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            # Track which source queries belong to this batch so we can map
            # fallback values back to the correct index after retries.
            batch_offset = len(translations)
            batch_queries = queries[batch_offset:batch_offset + query_size]

            language_retry_attempt = 0
            request_prompt = prompt
            self._vi_retry_extra_system = ''  # reset per prompt batch
            cleaned_translations: List[str] = []

            while True:
                self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, request_prompt))

                ratelimit_attempt = 0
                server_error_attempt = 0
                timeout_attempt = 0
                while True:
                    request_task = asyncio.create_task(self._request_translation(to_lang, request_prompt))
                    started = time.time()
                    while not request_task.done():
                        await asyncio.sleep(0.1)
                        if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                            # Server takes too long to respond
                            if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                                raise Exception('ollama servers did not respond quickly enough.')
                            timeout_attempt += 1
                            self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                            request_task.cancel()
                            request_task = asyncio.create_task(self._request_translation(to_lang, request_prompt))
                            started = time.time()
                    try:
                        response = await request_task
                        break
                    except openai.RateLimitError:  # Server returned ratelimit response
                        ratelimit_attempt += 1
                        if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                            raise
                        self.logger.warning(
                            f'Restarting request due to ratelimiting by Ollama servers. Attempt: {ratelimit_attempt}')
                        await asyncio.sleep(2)
                    except openai.APIError:  # Server returned 500 error (probably server load)
                        server_error_attempt += 1
                        if server_error_attempt >= self._RETRY_ATTEMPTS:
                            self.logger.error(
                                'Ollama encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                            raise
                        self.logger.warning(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                        await asyncio.sleep(1)

                # Use regex to extract response 
                response = self.extract_capture_groups(response, rf"{self.rgx_capture}") or ""
                response = _strip_generation_artifacts(response, preserve_segment_tokens=True)


                # Sometimes it will return line like "<|9>demo", and we need to fix it.
                def add_pipe(match):
                    number = match.group(1)
                    return f"<|{number}|>"
                response = re.sub(r"<\|?(\d+)\|?>", add_pipe, response)

                # Convert literal escaped newlines into real newlines so rendered text doesn't show "\\n" or "/n"
                while '\\n' in response or '\\r' in response or '/n' in response or '/r' in response:
                    response = response.replace('\\r\\n', '\n')
                    response = response.replace('\\n', '\n')
                    response = response.replace('\\r', '\n')
                    response = response.replace('/r/n', '\n')
                    response = response.replace('/n', '\n')
                    response = response.replace('/r', '\n')

                # Remove any text preceeding the first translation.
                new_translations = re.split(r'<\|[^|]+\|>', 'pre_1\n' + response)[1:]

                # When there is only one query LLMs likes to exclude the <|1|>
                if not new_translations:
                    new_translations = [response]

                # Immediately clean leading and trailing whitespace from each translation text
                cleaned_translations = [_clean_watermark_fragments(_strip_generation_artifacts(t)).strip() for t in new_translations]

                if len(cleaned_translations) <= 1 and query_size > 1:
                    # Try splitting by newlines instead
                    cleaned_translations = [
                        _clean_watermark_fragments(_strip_generation_artifacts(t)).strip()
                        for t in re.split(r'\n', response)
                    ]

                if len(cleaned_translations) > query_size:
                    cleaned_translations = cleaned_translations[: query_size]
                elif len(cleaned_translations) < query_size:
                    cleaned_translations = cleaned_translations + [''] * (query_size - len(cleaned_translations))

                if _target_is_vietnamese(to_lang):
                    non_vietnamese = [t for t in cleaned_translations if _needs_vietnamese_retry(t)]
                    if non_vietnamese and language_retry_attempt < 10:
                        language_retry_attempt += 1
                        self.logger.warning(
                            f'Retrying because output is not fully Vietnamese. Attempt: {language_retry_attempt}'
                        )
                        # Inject retry instruction into SYSTEM message, NOT user prompt.
                        # Injecting into user prompt causes weak models to echo the
                        # instructions as translation content, rendering them onto the image.
                        self._vi_retry_extra_system = self._get_vietnamese_retry_system_suffix()
                        request_prompt = prompt  # keep user message clean
                        continue

                self._vi_retry_extra_system = ''  # clear after successful pass
                break

            # ── Post-retry fallback for Vietnamese target ──────────────────────────────
            # Applied AFTER all inner Vietnamese retries are exhausted.
            #
            # • Watermark source queries → replace with U+200D (ZERO WIDTH JOINER):
            #     - '\u200d'.strip() == '\u200d'  (non-empty → MIT keeps region)
            #     - Region is inpainted (watermark area erased from image)
            #     - ZWJ has no visible glyph → nothing is rendered in its place
            #     - Fixes Issue 3a: watermarks are now actually erased
            #
            # • Content queries still returning English → fall back to source text:
            #     - MIT sees translation == original → "Translation identical to original"
            #     - Region is NOT inpainted; original Chinese text stays visible
            #     - Better than rendering English text on the image
            #     - Fixes Issue 1 + Issue 3b (slot stays occupied, no misalignment)
            if _target_is_vietnamese(to_lang):
                for i in range(len(cleaned_translations)):
                    src = batch_queries[i] if i < len(batch_queries) else ""
                    if _contains_watermark_text(src):
                        # Always erase watermarks regardless of what model returned.
                        cleaned_translations[i] = "\u200d"
                    elif _needs_vietnamese_retry(cleaned_translations[i]):
                        # Translation still English after all retries.
                        # Revert to source so MIT preserves the original Chinese text.
                        self.logger.warning(
                            f'Segment {i} not Vietnamese after retries; '
                            f'reverting to source text to preserve original.'
                        )
                        cleaned_translations[i] = src if src else cleaned_translations[i]

            translations.extend(cleaned_translations)

        for t in translations:
            if "I'm sorry, but I can't assist with that request" in t:
                raise Exception('translations contain error text')
        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        sys_content = self.chat_system_template.format(to_lang=to_lang)
        if getattr(self, '_vi_retry_extra_system', ''):
            sys_content += self._vi_retry_extra_system
        messages = [{'role': 'system', 'content': sys_content}]

        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        messages.append({'role': 'user', 'content': prompt})

        response = await self.client.chat.completions.create(
            model=self.model or CUSTOM_OPENAI_MODEL,
            messages=messages,
            max_tokens=self._MAX_TOKENS,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        self.logger.debug('\n-- GPT Response (raw) --')
        self.logger.debug(response.choices[0].message.content)
        self.logger.debug('------------------------\n')


        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        return response.choices[0].message.content
