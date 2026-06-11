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
_TYPE_TAG_RE = re.compile(r"^\s*\[\s*(speech|thought|moan|shout|narration|sfx|anger|fear|title)\s*\]\s*", re.IGNORECASE)

# Nhãn [cont]: segment là PHẦN SAU của một câu nguồn bị detector cắt thành nhiều
# vùng — bản dịch TRỌN câu nằm ở segment anchor phía trước. Giờ chỉ là LƯỚI AN TOÀN
# (model tự phát ra thì vẫn xử lý đúng); việc gộp câu chính thức do pre-pass
# _plan_sentence_merges đảm nhiệm. Bắt cả biến thể [continuation]/[continued]/[cont.].
_CONT_TAG_RE = re.compile(r"^\s*\[\s*cont[\w.]*\s*\]\s*", re.IGNORECASE)

# Dấu KẾT câu CJK/Latin: segment kết thúc bằng những dấu này coi như TRỌN câu —
# KHÔNG phải ứng viên "bị cắt giữa chừng" cho pre-pass gộp câu.
_CJK_TERMINAL_PUNCT = set('。．.!！?？…⋯~～—–-"\'”’」』】)）]')

# Cảnh giới tu tiên: map zh→vn để ENFORCE sau dịch. Model hay trượt (vd 结丹初期 →
# "Trúc Cơ sơ kỳ") vì story context các trang trước nhắc "Trúc Cơ" liên tục — prompt
# không đủ giữ. Vocabulary đóng nên sửa deterministic được.
_REALM_MAP = [
    ('炼气', 'Luyện Khí'), ('练气', 'Luyện Khí'),
    ('筑基', 'Trúc Cơ'),
    ('结丹', 'Kết Đan'), ('結丹', 'Kết Đan'),
    ('金丹', 'Kim Đan'),
    ('元婴', 'Nguyên Anh'), ('元嬰', 'Nguyên Anh'),
    ('化神', 'Hóa Thần'),
    ('炼虚', 'Luyện Hư'), ('煉虛', 'Luyện Hư'),
    ('合体', 'Hợp Thể'), ('合體', 'Hợp Thể'),
    ('大乘', 'Đại Thừa'),
    ('渡劫', 'Độ Kiếp'),
]


# Lỗi OCR phổ biến ở THUẬT NGỮ tu tiên: chuỗi bên trái KHÔNG phải từ thật trong
# tiếng Trung — chỉ có thể là OCR nhầm glyph (丹→母…). Sửa trong BẢN GỬI LLM để
# model dịch đúng ngay (key store/region.text vẫn dùng nguyên văn OCR gốc).
_OCR_SOURCE_FIX = [
    ('结母', '结丹'), ('結母', '結丹'),
]

# OCR đọc dấu lửng "……" thành chuỗi chữ 'o' (vd "非常好oooo那" / "夫人吧oo"):
# run [oO0]{2,} dính liền chữ Hán KHÔNG bao giờ là từ thật → thay bằng "……" để
# model dịch ra "…" thay vì bê nguyên "oooo" vào bản dịch ("tốtoooo", "đi oo.").
# Điều kiện kẹp CJK giữ an toàn cho watermark/từ Latin thật ("gxracg.com", "cool"):
#   - đứng SAU chữ Hán và KHÔNG nối tiếp bằng chữ/số Latin (cuối chuỗi, trước CJK
#     hay trước dấu câu đều khớp), HOẶC
#   - mở ĐẦU chuỗi và nối thẳng vào chữ Hán.
_CJK_CLASS = r'一-鿿㐀-䶿'
_OCR_ELLIPSIS_RE = re.compile(
    rf'(?<=[{_CJK_CLASS}])[oO0]{{2,}}(?![A-Za-z0-9])'
    rf'|^[oO0]{{2,}}(?=[{_CJK_CLASS}])'
)


def _fix_ocr_source(q: str) -> str:
    if not isinstance(q, str) or not q:
        return q
    for wrong, right in _OCR_SOURCE_FIX:
        if wrong in q:
            q = q.replace(wrong, right)
    q = _OCR_ELLIPSIS_RE.sub('……', q)
    return q


def _fix_realm_terms(src: str, trans: str, logger=None) -> str:
    """Sửa tên CẢNH GIỚI dịch sai: nguồn có cảnh giới X mà bản dịch (a) THIẾU tên
    đúng của X và (b) lại chứa tên cảnh giới KHÁC không hề có trong nguồn → thay
    tên sai bằng tên đúng. Chỉ đụng khi cả 2 điều kiện cùng thoả → an toàn với các
    câu nhắc nhiều cảnh giới thật (vd thuốc 筑基丹 trong câu nói về 结丹)."""
    if not src or not trans or not isinstance(trans, str):
        return trans
    src_pairs = [(zh, vn) for zh, vn in _REALM_MAP if zh in src]
    if not src_pairs:
        return trans
    src_vn_names = {vn for _, vn in src_pairs}
    out = trans
    for zh, vn in src_pairs:
        if vn.lower() in out.lower():
            continue  # bản dịch đã có tên đúng
        for _, wrong_vn in _REALM_MAP:
            if wrong_vn in src_vn_names:
                continue  # tên này hợp lệ (nguồn có cảnh giới đó)
            m = re.search(re.escape(wrong_vn), out, re.IGNORECASE)
            if m:
                out = out[:m.start()] + vn + out[m.end():]
                if logger:
                    logger.info(f'[realm-fix] "{wrong_vn}" → "{vn}" (nguồn có {zh}).')
                break
    return out


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


def _expand_merged_translations(orig_queries: List[str], groups: List[List[int]],
                                translations: List[str], logger) -> List[str]:
    """Trải bản dịch của các nhóm ĐÃ GỘP (pre-pass merge) về đúng slot vùng gốc.

    Anchor (segment đầu nhóm) nhận bản dịch trọn câu; các segment sau nhận ZWJ
    (vô hình — vùng vẫn được inpaint, không render) + ghi map cont→anchor để
    renderer UNION box. Nhãn loại thoại được lưu theo text GỘP trong lúc dịch →
    copy về key text gốc của anchor (renderer tra theo region.text vùng gốc)."""
    if len(translations) < len(groups):
        translations = translations + [''] * (len(groups) - len(translations))
    _merges = _region_merge_store()
    _types = _region_type_store()
    expanded = ['‍'] * len(orig_queries)
    for g, tr in zip(groups, translations):
        a = g[0]
        anchor_key = (orig_queries[a] or '').strip()
        if len(g) > 1:
            merged_key = ''.join((orig_queries[j] or '').strip() for j in g)
            if (tr or '').strip() == merged_key:
                # Fallback "giữ nguyên gốc" ([title] nghi nhầm / không dịch được):
                # trả MỖI vùng đúng text gốc CỦA NÓ → MIT thấy dịch == gốc, bỏ qua
                # inpaint, chữ thư pháp gốc còn nguyên. KHÔNG ghi map merge — nếu
                # ghi, anchor nhận chuỗi GỘP (≠ text vùng) sẽ bị inpaint rồi render
                # lại chữ Hán bằng font Việt, còn vùng cont bị xoá trắng.
                for j in g:
                    expanded[j] = orig_queries[j]
                logger.info(f'[merge-llm] segments {[k + 1 for k in g]} giữ nguyên '
                            f'chữ gốc (không dịch) — bỏ union, không inpaint.')
                continue
        expanded[a] = tr
        if len(g) > 1:
            if anchor_key and merged_key in _types:
                _types[anchor_key] = _types[merged_key]
            for j in g[1:]:
                ck = (orig_queries[j] or '').strip()
                if ck and anchor_key and ck != anchor_key:
                    _merges[ck] = anchor_key
            logger.info(f'[merge-llm] segments {[k + 1 for k in g]} là MỘT câu — '
                        f'dịch gộp, union box khi render.')
    return expanded
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


# Dấu hiệu TIÊU ĐỀ/TRANG TRÍ thật: Latin/số (cả fullwidth), ngoặc, marker chương
# (第/章/话/回/卷/期). Chuỗi Hán THUẦN không có các dấu hiệu này có thể là LỜI DẪN
# viết thư pháp bị model dán nhầm nhãn [title] — không được phép xoá.
_TITLE_DECOR_RE = re.compile(
    r"[A-Za-z0-9０-９Ａ-Ｚａ-ｚ()（）\[\]【】"
    r"《》〈〉#＃*＊·•:：]"
    r"|第|章|话|話|回|卷|期"
)


def _title_safe_to_erase(text: str) -> bool:
    """[title] chỉ CHẮC CHẮN là trang trí (được xoá khỏi ảnh) khi nguồn mang dấu
    hiệu tiêu đề: Latin/số/ngoặc/số chương, hoặc chuỗi rác OCR rất dài. Câu Hán
    thuần ngắn (vd "于逆星之下举起义旗") có thể là lời dẫn cảnh bị phân loại nhầm
    → trả False: GIỮ NGUYÊN chữ gốc thay vì xoá mất nội dung truyện."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) >= 16:  # chuỗi rác OCR dài — lời dẫn thật hiếm khi dài cỡ này mà bị gán [title]
        return True
    return bool(_TITLE_DECOR_RE.search(t))


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
            prompt += f'\n<|{i + 1 - i_offset}|>{_fix_ocr_source(query)}'

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
        _region_merge_store().clear()  # map gộp câu cũng chỉ cho trang hiện tại

        # ── Pre-pass GỘP CÂU bằng LLM (semantic) ────────────────────────────────
        # Các segment là MỘT câu bị detector cắt giữa chừng được GỘP TRƯỚC khi dịch
        # → model dịch trọn câu một lần (slot anchor), các slot sau nhận ZWJ + map
        # cho renderer union box. Thay cho contract [cont] in-band: model yếu hay
        # echo ví dụ trong rule / bỏ qua nhãn, còn task YES/NO riêng thì trả ổn.
        orig_queries = list(queries)
        groups = [[i] for i in range(len(queries))]
        try:
            groups = await asyncio.wait_for(self._plan_sentence_merges(queries), timeout=45)
        except Exception as e:
            self.logger.warning(f'[merge-llm] pre-pass lỗi/timeout ({e}) — dịch từng segment như cũ.')
            groups = [[i] for i in range(len(orig_queries))]
        has_merge = any(len(g) > 1 for g in groups)
        if has_merge:
            queries = [''.join((orig_queries[j] or '').strip() for j in g) for g in groups]
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
                    elif _store.get(src.strip()) == "title":
                        # [title] = ti\u00eau \u0111\u1ec1/ch\u1eef trang tr\u00ed/con d\u1ea5u \u2014 OCR c\u1ee7a ch\u1eef c\u00e1ch
                        # \u0111i\u1ec7u th\u01b0\u1eddng l\u00e0 chu\u1ed7i r\u00e1c, "b\u1ea3n d\u1ecbch" ch\u1ec9 ra ch\u1eef to v\u00f4 ngh\u0129a
                        # \u0111\u00e8 artwork \u2192 xo\u00e1 nh\u01b0 watermark, k\u1ec3 c\u1ea3 khi model l\u1ee1 k\u00e8m d\u1ecbch.
                        # NH\u01afNG model hay d\u00e1n nh\u1ea7m [title] cho L\u1edcI D\u1eaaN vi\u1ebft th\u01b0 ph\u00e1p
                        # (vd "\u4e8e\u9006\u661f\u4e4b\u4e0b\u4e3e\u8d77\u4e49\u65d7" b\u1ecb xo\u00e1 tr\u1eafng m\u1ea5t n\u1ed9i dung truy\u1ec7n).
                        # Ngu\u1ed3n KH\u00d4NG c\u00f3 d\u1ea5u hi\u1ec7u ti\u00eau \u0111\u1ec1 \u2192 GI\u1eee NGUY\u00caN ch\u1eef g\u1ed1c tr\u00ean
                        # \u1ea3nh (thi\u1ebfu b\u1ea3n d\u1ecbch v\u1eabn h\u01a1n xo\u00e1 m\u1ea5t ch\u1eef).
                        if _title_safe_to_erase(src):
                            cleaned_translations[i] = "\u200d"
                        else:
                            self.logger.warning(
                                f'[title] cho segment {i} nh\u01b0ng ngu\u1ed3n "{src.strip()[:20]}" '
                                f'tr\u00f4ng nh\u01b0 c\u00e2u c\u00f3 ngh\u0129a \u2014 gi\u1eef ch\u1eef g\u1ed1c, kh\u00f4ng xo\u00e1.')
                            cleaned_translations[i] = src
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
                    # Enforce t\u00ean c\u1ea3nh gi\u1edbi kh\u1edbp ngu\u1ed3n (sau m\u1ecdi x\u1eed l\u00fd kh\u00e1c). Soi tr\u00ean
                    # ngu\u1ed3n \u0110\u00c3 s\u1eeda l\u1ed7i OCR (\u7ed3\u6bcd\u2192\u7ed3\u4e39) \u2014 kh\u1edbp v\u1edbi b\u1ea3n model nh\u00ecn th\u1ea5y.
                    if cleaned_translations[i] not in ("", "\u200d"):
                        cleaned_translations[i] = _fix_realm_terms(
                            _fix_ocr_source(src), cleaned_translations[i], self.logger)

            translations.extend(cleaned_translations)

        # Trải kết quả các nhóm ĐÃ GỘP về đúng slot vùng gốc.
        if has_merge:
            translations = _expand_merged_translations(
                orig_queries, groups, translations, self.logger)

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

    async def _plan_sentence_merges(self, queries: List[str]) -> List[List[int]]:
        """Nhóm các segment nguồn là MỘT câu bị detector cắt giữa chừng (pre-pass).

        Heuristic rẻ chọn ỨNG VIÊN: segment CJK (không phải watermark) kết thúc
        KHÔNG có dấu kết câu + segment kế tiếp cũng là CJK thường. Rồi MỘT call
        LLM riêng chỉ hỏi YES/NO từng cặp — task nhỏ, output ràng buộc nên model
        yếu trả lời ổn định (khác với contract [cont] nhét trong lúc dịch: model
        hay echo ví dụ hoặc bỏ qua nhãn). Mọi lỗi/không chắc → singleton (như cũ).

        Trả về list nhóm chỉ số gốc, vd [[0],[1],[2,3],[4]] = segment 3+4 là một câu."""
        n = len(queries)
        singles = [[i] for i in range(n)]
        if n < 2:
            return singles
        cands = []
        for i in range(n - 1):
            # Soi trên bản ĐÃ sửa lỗi OCR: "吧oo" (OCR của "吧……") phải được nhận
            # diện là đã KẾT câu, không thành ứng viên gộp với segment sau.
            a = _fix_ocr_source((queries[i] or '').strip())
            b = _fix_ocr_source((queries[i + 1] or '').strip())
            if not a or not b:
                continue
            if not _CHINESE_RE.search(a) or not _CHINESE_RE.search(b):
                continue
            if _contains_watermark_text(a) or _contains_watermark_text(b):
                continue
            if a[-1] in _CJK_TERMINAL_PUNCT:
                continue
            # Nhãn/danh xưng NGẮN (≤4 chữ CJK: 白面儒生, 精壮大汉, 结丹初期…) là
            # caption độc lập đứng cạnh vùng khác — câu thật bị detector cắt hầu
            # như luôn dài hơn. Loại khỏi ứng viên để LLM không có cơ hội gộp nhầm.
            if sum(1 for c in a if _CHINESE_RE.match(c)) <= 4:
                continue
            cands.append(i)
        if not cands:
            return singles
        qlines = []
        for k, i in enumerate(cands, 1):
            qlines.append(f'{k}. A=「{_fix_ocr_source((queries[i] or "").strip())}」'
                          f' B=「{_fix_ocr_source((queries[i + 1] or "").strip())}」')
        resp = await self._request_merge_plan('\n'.join(qlines))
        confirmed = set()
        for k, i in enumerate(cands, 1):
            m = re.search(rf'^\s*{k}\s*[:.)\-]?\s*(YES|NO)\b', resp or '', re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).upper() == 'YES':
                confirmed.add(i)
        if not confirmed:
            return singles
        groups = []
        i = 0
        while i < n:
            g = [i]
            while g[-1] in confirmed:
                g.append(g[-1] + 1)
            groups.append(g)
            i = g[-1] + 1
        return groups

    async def _request_merge_plan(self, pairs_block: str) -> str:
        """Call LLM TỐI GIẢN (system riêng, không dính prompt dịch) trả YES/NO cho
        từng cặp segment nghi là một câu bị cắt. Few-shot có cả ca NO (tên + chức
        danh) để model không gộp bừa các cột caption đứng cạnh nhau."""
        sys_content = (
            "Bạn là bộ phân tích văn bản OCR truyện tranh Trung văn. Mỗi dòng có cặp "
            "đoạn A và B (B đứng NGAY SAU A trên trang). Trả lời YES CHỈ KHI A bị CẮT "
            "GIỮA CHỪNG (chưa trọn câu, ý đang dở, ngắt giữa cụm từ) và B nối thẳng "
            "vào A thành một câu liền mạch. Trả lời NO nếu A và B là hai câu/cụm độc "
            "lập. Đặc biệt LUÔN là NO khi: A là nhãn/danh xưng/miêu tả ngắn về nhân "
            "vật (tên, chức danh, cảnh giới), HOẶC B tự nó đã là một câu hoàn chỉnh "
            "đứng riêng được. Phân vân → NO.\n"
            "CHỈ trả lời mỗi dòng dạng 'số: YES' hoặc 'số: NO'. Không giải thích."
        )
        messages = [
            {'role': 'system', 'content': sys_content},
            {'role': 'user', 'content': '1. A=「他拿起手中的」 B=「剑朝我刺来。」\n'
                                        '2. A=「李雷」 B=「高级炼金术师」\n'
                                        '3. A=「白面书生哥哥」 B=「圣女被困外海，正被两个修士追杀！」'},
            {'role': 'assistant', 'content': '1: YES\n2: NO\n3: NO'},
            {'role': 'user', 'content': pairs_block},
        ]
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        response = await self.client.chat.completions.create(
            model=self.model or CUSTOM_OPENAI_MODEL,
            messages=messages,
            max_tokens=256,
            temperature=0,
            top_p=1.0,
            seed=self._SEED if self._DETERMINISTIC else None,
            extra_body=extra_body,
        )
        try:
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens
        except Exception:
            pass
        out = response.choices[0].message.content or ''
        self.logger.info(f'[merge-llm] plan ({pairs_block.count(chr(10)) + 1} cặp):\n{out.strip()}')
        return out

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
