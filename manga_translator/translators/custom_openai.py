import re
from collections import deque

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

# Nh\u00e3n lo\u1ea1i tho\u1ea1i do model g\u00e1n \u1edf \u0111\u1ea7u m\u1ed7i \u0111o\u1ea1n: "[thought] ...". B\u00f3c ra \u0111\u1ec3 (a) kh\u00f4ng
# l\u1ecdt v\u00e0o b\u1ea3n d\u1ecbch hi\u1ec3n th\u1ecb, (b) kh\u00f4ng b\u1ecb b\u1ed9 check ti\u1ebfng Vi\u1ec7t t\u01b0\u1edfng l\u00e0 ti\u1ebfng Anh.
_TYPE_TAG_RE = re.compile(r"^\s*\[\s*(speech|thought|moan|shout|narration|sfx|anger|fear)\s*\]\s*", re.IGNORECASE)

# Nhãn [cont]: segment là PHẦN SAU của một câu nguồn bị detector cắt thành nhiều
# vùng — bản dịch TRỌN câu nằm ở segment anchor phía trước (rule 6c system prompt).
# Bắt cả biến thể [continuation]/[continued]/[cont.].
_CONT_TAG_RE = re.compile(r"^\s*\[\s*cont[\w.]*\s*\]\s*", re.IGNORECASE)


def _region_type_store() -> dict:
    """Dict d\u00f9ng chung gi\u1eefa translator v\u00e0 renderer (c\u00f9ng ti\u1ebfn tr\u00ecnh MIT), kho\u00e1 =
    text g\u1ed1c (CJK) \u0111\u00e3 strip \u2192 lo\u1ea1i tho\u1ea1i. Renderer tra theo region.text \u0111\u1ec3 ch\u1ecdn font."""
    import manga_translator as _mt
    d = getattr(_mt, "_VI_REGION_TYPES", None)
    if d is None:
        d = {}
        _mt._VI_REGION_TYPES = d
    return d


def _region_merge_store() -> dict:
    """Dict dùng chung translator↔renderer (cùng tiến trình MIT): text gốc của vùng
    [cont] → text gốc của vùng anchor. Renderer dùng để UNION box các vùng vốn là
    MỘT câu bị detector cắt đôi. Xoá mỗi trang (như _region_type_store)."""
    import manga_translator as _mt
    d = getattr(_mt, "_VI_REGION_MERGES", None)
    if d is None:
        d = {}
        _mt._VI_REGION_MERGES = d
    return d
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
        # Sót pipe LẺ từ marker méo (vd model xuất '<||3|>' → split để lại '|', hoặc
        # model tự gõ '|'). '|' KHÔNG bao giờ là ký tự hợp lệ trong thoại tiếng Việt →
        # bỏ hẳn. Nếu KHÔNG bỏ, '|' lọt xuống _ensure_terminal_punct (không phải dấu
        # kết, không phải ngoặc đóng) → bị nối thêm '.' → ra "「Phạm phu nhân」|.".
        cleaned = cleaned.replace("|", "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Bảo đảm thoại kết thúc bằng dấu câu ──────────────────────────────────────
# Dấu KẾT câu hợp lệ: nếu thoại đã kết thúc bằng một trong số này thì GIỮ NGUYÊN.
# Gồm cả dấu fullwidth (CJK), dấu lửng "…" và gạch ngang "—" (thoại bị ngắt/cắt
# ngang — thêm '.' vào đây sẽ phá ý đồ). ',;:' cũng đã là dấu câu nên không thêm.
_TERMINAL_PUNCT = set(".!?…~" "。！？⋯～" "—–-" ",;:")
# Ký tự ĐÓNG (ngoặc / nháy) có thể đứng SAU dấu kết câu, vd: 「Xin chào!」 — nhìn
# xuyên qua chúng để xét dấu câu THẬT bên trong.
_TRAILING_CLOSERS = set('"\'”’»)]}）】」』›')


def _ensure_terminal_punct(text: str) -> str:
    """Bảo đảm thoại kết thúc bằng dấu câu (mặc định thêm '.').

    Bỏ qua: chuỗi rỗng, marker ZWJ (vùng watermark đã xoá) và chuỗi chỉ gồm ký tự
    đóng. SFX (tượng thanh: rầm, bùm…) KHÔNG phải thoại nên được caller loại trừ
    trước khi gọi hàm này."""
    if not isinstance(text, str):
        return text
    stripped = text.rstrip()
    if not stripped or stripped == "‍":
        return text
    # Nhìn xuyên qua ký tự đóng ở cuối để xét dấu câu thật bên trong.
    idx = len(stripped)
    while idx > 0 and stripped[idx - 1] in _TRAILING_CLOSERS:
        idx -= 1
    if idx == 0:
        return stripped  # chỉ toàn ký tự đóng — không đụng tới
    if stripped[idx - 1] in _TERMINAL_PUNCT:
        return stripped  # đã có dấu kết câu
    return stripped[:idx] + "." + stripped[idx:]


# Chỉ-toàn-rác: ngoặc/dấu câu/marker/khoảng trắng, KHÔNG có chữ-số thật. Dùng để bắt
# các segment model trả rỗng kiểu "[]", "[ ]", "[.]", "【】", "..." → coi là TRỐNG.
_MEANINGLESS_RE = re.compile(
    r"[\s\[\](){}<>|.,!?…~。！？⋯—–\-:;\"'`“”‘’«»「」『』【】（）]+"
)


def _is_effectively_empty(text: str) -> bool:
    """True nếu segment KHÔNG còn nội dung thật sau khi bỏ nhãn loại + mọi dấu/ngoặc.
    Bắt các trường hợp model trả "[]" / "[.]" / "【】" (rỗng có chủ ý) — nếu KHÔNG bắt,
    "[]" lọt xuống _ensure_terminal_punct → bị chèn '.' thành "[.]" → render ra rác."""
    if not isinstance(text, str):
        return True
    t = _TYPE_TAG_RE.sub("", text.strip())   # bỏ "[speech]"… nếu có
    t = _MEANINGLESS_RE.sub("", t)
    return t == "" or t == "‍"


def _contains_watermark_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    compact = re.sub(r"\s+", "", text.lower())
    compact = re.sub(r"[^a-z0-9._:/-]", "", compact)
    # Site / handle / domain tags that only ever appear as scanlation watermarks.
    # Pixiv ID watermark, kể cả khi OCR đọc lệch I↔1↔l (vd "PIX1V:fh8di",
    # "P1XIV:...", "P1X1V:..."). Tag "pixiv" thuần ở dưới bắt trượt các biến thể này.
    if re.search(r"p[i1l]x[i1l]v", compact):
        return True
    site_tags = (
        "acg",
        "pixiv",
        "twitter",
        "weibo",
        "fanbox",
        "patreon",
        "fantia",
        "danbooru",
        "dlsite",
        "bilibili",
    )
    return (
        any(tag in compact for tag in site_tags)
        or ".com" in compact
        or ".net" in compact
        or ".org" in compact
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
    # Số thứ tự lần retry tiếng Việt hiện tại (0 = lần đầu). Dùng để lệch seed mỗi
    # lần retry → output thật sự KHÁC nhau, model có cơ hội dịch nốt chữ Hán còn sót
    # (seed cố định khiến 10 lần retry ra y hệt, vô dụng). Lần đầu vẫn seed cố định
    # ⇒ giữ tính tái lập cho ca thường.
    _vi_retry_attempt: int = 0

    # ── Cross-page story context ─────────────────────────────────────────────
    # MIT reuses one translator instance for every image in a run, so we keep a
    # rolling buffer of the most recent translated pages. It is injected into the
    # system prompt as read-only reference, letting the model keep names/pronouns/
    # tone consistent and pick scene-appropriate wording across the chapter.
    _CONTEXT_MAX_PAGES = 10    # nhớ nội dung ~10 ảnh gần nhất
    _CONTEXT_MAX_CHARS = 1200  # giới hạn cứng độ dài context bơm vào prompt (tránh phình token)

    # ── Sampling — NGUỒN SỰ THẬT DUY NHẤT ────────────────────────────────
    # Quyết định sampling tập trung ở ĐÂY, ghi đè mọi temperature/top_p từ
    # gpt_config_vi.yaml và config tạm theo style.
    #
    # MẤU CHỐT: tái lập (mỗi lần refresh ra cùng kết quả) đến từ SEED CỐ ĐỊNH,
    # KHÔNG phải từ greedy. llama.cpp/LM Studio: cùng seed + cùng params + cùng
    # prompt → cùng output, kể cả ở temperature 0.7. Ép greedy (temp=0/top_k=1)
    # làm model sáng tạo như Qwen dịch cụt, phẳng, rớt nghĩa — nên KHÔNG dùng.
    # Bộ tham số dưới là khuyến nghị của Qwen3 (chế độ non-thinking) để dịch
    # ngon NGANG khung chat, mà vẫn tái lập nhờ seed.
    #   _DETERMINISTIC=True  (mặc định) → sampling chất lượng + seed cố định.
    #   _DETERMINISTIC=False → dùng temperature/top_p từ config, seed ngẫu nhiên.
    _DETERMINISTIC = True
    _SEED = 1234
    _SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                 "min_p": 0.0, "repeat_penalty": 1.0}

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
        # Rolling buffer of recent translated pages (each item = list of VI lines).
        self._recent_context: deque = deque(maxlen=self._CONTEXT_MAX_PAGES)

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

    def _build_context_block(self) -> str:
        """Read-only story context from recently translated pages.

        Injected into the SYSTEM message (not the user prompt, to avoid weak
        models echoing it as translation content). Helps the model keep names,
        pronouns, tone and vocabulary consistent and choose words that fit the
        ongoing scene across the whole chapter.
        """
        if not self._recent_context:
            return ''
        flat: List[str] = []
        for page in self._recent_context:
            flat.extend(page)
        text = ' / '.join(s.replace('\n', ' ').strip() for s in flat if s and s.strip())
        text = text.strip()
        if not text:
            return ''
        if len(text) > self._CONTEXT_MAX_CHARS:
            # Keep the most recent tail — closest to the current scene.
            text = '…' + text[-self._CONTEXT_MAX_CHARS:]
        return (
            "\n\nSTORY CONTEXT (recent dialogue from previous pages, Vietnamese — "
            "REFERENCE ONLY). Use it to keep character names, pronouns, tone and "
            "vocabulary consistent, and to choose words that fit the ongoing scene. "
            "NEVER translate, repeat or output these lines — only output the <|n|> "
            "segments for the new text:\n" + text
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
        _region_type_store().clear()   # nhãn loại thoại chỉ cho trang hiện tại
        _region_merge_store().clear()  # map gộp [cont] cũng chỉ cho trang hiện tại
        if self._DETERMINISTIC:
            self.logger.debug(f'Sampling: reproducible ({self._SAMPLING}, seed={self._SEED})')
        else:
            self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size in self._assemble_prompts(from_lang, to_lang, queries):
            # Track which source queries belong to this batch so we can map
            # fallback values back to the correct index after retries.
            batch_offset = len(translations)
            batch_queries = queries[batch_offset:batch_offset + query_size]

            language_retry_attempt = 0
            request_prompt = prompt
            self._vi_retry_extra_system = ''  # reset per prompt batch
            self._vi_retry_attempt = 0       # lần đầu dùng seed cố định
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

                # ── Bóc nhãn loại thoại "[type]" ở đầu mỗi đoạn ──────────────────
                # Lưu (text gốc → loại) cho renderer chọn font, rồi BỎ nhãn khỏi bản
                # dịch (trước khi check tiếng Việt, kẻo "[moan] Haa" bị tưởng tiếng Anh).
                _store = _region_type_store()
                for _i in range(len(cleaned_translations)):
                    _m = _TYPE_TAG_RE.match(cleaned_translations[_i] or "")
                    if _m:
                        cleaned_translations[_i] = cleaned_translations[_i][_m.end():].lstrip()
                        _src = batch_queries[_i].strip() if _i < len(batch_queries) else ""
                        if _src:
                            _store[_src] = _m.group(1).lower()

                # ── Bóc nhãn [cont]: câu nguồn bị detector cắt thành nhiều segment ──
                # Model dịch TRỌN câu vào segment anchor (đầu chuỗi) và trả "[cont]"
                # cho các segment tiếp theo (rule 6c). Ghi map (text cont → text
                # anchor) để renderer UNION box các vùng, rồi thay segment cont bằng
                # ZWJ — placeholder vô hình đã được mọi bộ check chấp nhận ('' sẽ
                # kích retry "translation invalid"). Nhờ map này các bản dịch phía
                # sau KHÔNG còn bị dồn lệch slot như khi model tự ý gộp.
                _merges = _region_merge_store()
                _anchor_i = None
                for _i in range(len(cleaned_translations)):
                    _t = cleaned_translations[_i] or ""
                    _cm = _CONT_TAG_RE.match(_t)
                    if _cm is not None:
                        _rest = _t[_cm.end():].strip()
                        if _rest or _anchor_i is None:
                            # "[cont] chữ…" (model lệch contract) hoặc [cont] mở đầu
                            # không có anchor → bỏ nhãn, giữ phần còn lại như bản
                            # dịch thường (rỗng thì thay ZWJ để khỏi kích retry).
                            cleaned_translations[_i] = _rest if _rest else "‍"
                            if _rest:
                                _anchor_i = _i
                            continue
                        _src = batch_queries[_i].strip() if _i < len(batch_queries) else ""
                        _anchor_src = batch_queries[_anchor_i].strip() if _anchor_i < len(batch_queries) else ""
                        if _src and _anchor_src and _src != _anchor_src:
                            _merges[_src] = _anchor_src
                            self.logger.info(f'[cont] segment {_i + 1} gộp vào segment {_anchor_i + 1} '
                                             f'(một câu bị cắt đôi).')
                        cleaned_translations[_i] = "‍"
                        continue
                    if _t.strip() and _t.strip() != "‍":
                        _anchor_i = _i

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
                        self._vi_retry_attempt = language_retry_attempt  # lệch seed lần này
                        request_prompt = prompt  # keep user message clean
                        continue

                self._vi_retry_extra_system = ''  # clear after successful pass
                self._vi_retry_attempt = 0
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
                _store = _region_type_store()
                for i in range(len(cleaned_translations)):
                    src = batch_queries[i] if i < len(batch_queries) else ""
                    if _contains_watermark_text(src):
                        # Always erase watermarks regardless of what model returned.
                        cleaned_translations[i] = "\u200d"
                    elif _is_effectively_empty(cleaned_translations[i]):
                        # Model tr\u1ea3 r\u1ed7ng/ch\u1ec9-d\u1ea5u ("[]", "[.]", "\u3010\u3011"\u2026) \u2192 xo\u00e1 h\u1eb3n b\u1eb1ng ZWJ
                        # (v\u00f9ng \u0111\u01b0\u1ee3c inpaint, KH\u00d4NG render r\u00e1c). Tr\u00e1nh "[.]"\u2192"\u3010.\u3011" tr\u00ean \u1ea3nh.
                        cleaned_translations[i] = "\u200d"
                    elif _needs_vietnamese_retry(cleaned_translations[i]):
                        # Translation still English after all retries.
                        # Revert to source so MIT preserves the original Chinese text.
                        self.logger.warning(
                            f'Segment {i} not Vietnamese after retries; '
                            f'reverting to source text to preserve original.'
                        )
                        cleaned_translations[i] = src if src else cleaned_translations[i]
                    elif _store.get(src.strip()) != "sfx":
                        # Tho\u1ea1i (m\u1ecdi lo\u1ea1i TR\u1eea SFX t\u01b0\u1ee3ng thanh) ph\u1ea3i k\u1ebft th\u00fac b\u1eb1ng d\u1ea5u
                        # c\u00e2u. SFX (r\u1ea7m, b\u00f9m, v\u00fat\u2026) gi\u1eef nguy\u00ean \u2014 th\u00eam '.' s\u1ebd k\u1ef3 c\u1ee5c.
                        cleaned_translations[i] = _ensure_terminal_punct(cleaned_translations[i])

            translations.extend(cleaned_translations)

        for t in translations:
            if "I'm sorry, but I can't assist with that request" in t:
                raise Exception('translations contain error text')
        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        # Remember this page's real Vietnamese lines as context for later pages.
        # The VI-diacritic test alone filters out empties, ZWJ watermark markers
        # and source-text fallbacks (none of which carry Vietnamese diacritics).
        if _target_is_vietnamese(to_lang):
            page_vi = [
                t.strip() for t in translations
                if t and t.strip() and _VI_DIACRITIC_RE.search(t)
            ]
            if page_vi:
                self._recent_context.append(page_vi)

        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        sys_content = self.chat_system_template.format(to_lang=to_lang)
        context_block = self._build_context_block()
        if context_block:
            sys_content += context_block
        if getattr(self, '_vi_retry_extra_system', ''):
            sys_content += self._vi_retry_extra_system
        messages = [{'role': 'system', 'content': sys_content}]

        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        messages.append({'role': 'user', 'content': prompt})

        # Tắt "thinking" của Qwen3 → trả lời thẳng (content), không nhồi token vào
        # reasoning. Cần cho nhãn [type] ra sạch & ổn định.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        if self._DETERMINISTIC:
            # Sampling chất lượng (giống khung chat) + seed cố định ⇒ tái lập.
            # top_k/min_p/repeat_penalty là tham số riêng của llama.cpp/LM Studio
            # nên đi qua extra_body. repeat_penalty=1.0 (tắt) để KHÔNG phạt
            # marker <|n|> và SFX lặp (hà hà hà…), tránh méo bản dịch.
            s = self._SAMPLING
            temperature, top_p = s["temperature"], s["top_p"]
            # Lần đầu: seed cố định ⇒ tái lập. Khi đang retry tiếng Việt: BỎ seed
            # (None) để server tự random mỗi lần → output thật sự khác đi, cho model
            # cơ hội dịch nốt chữ Hán/Anh còn sót (seed cố định khiến retry ra y hệt).
            seed = None if self._vi_retry_attempt > 0 else self._SEED
            extra_body.update({"top_k": s["top_k"], "min_p": s["min_p"],
                               "repeat_penalty": s["repeat_penalty"]})
        else:
            temperature, top_p = self.temperature, self.top_p
            seed = None

        response = await self.client.chat.completions.create(
            model=self.model or CUSTOM_OPENAI_MODEL,
            messages=messages,
            max_tokens=self._MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            extra_body=extra_body,
        )

        self.logger.debug('\n-- GPT Response (raw) --')
        self.logger.debug(response.choices[0].message.content)
        self.logger.debug('------------------------\n')


        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        return response.choices[0].message.content
