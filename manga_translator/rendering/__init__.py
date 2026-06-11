import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from .text_render_pillow_eng import render_textblock_list_eng as render_textblock_list_eng_pillow
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

# ── Đa dạng font theo loại thoại ────────────────────────────────────────────
# Map loại thoại → tên file font (đặt CÙNG thư mục với font chính, vd fonts/).
# Thiếu file nào thì tự fallback về font chính (font_path) — an toàn.
# ⚠ Font phải có sẵn ký tự tiếng Việt (ố ữ ặ…); font Anh gốc sẽ ra ô vuông.
_FONT_BY_KIND = {
    "normal":    "MTO Astro City.ttf",   # thoại thường           (có sẵn)
    "thought":   "MTO COMIC 1.ttf",      # thoại suy nghĩ          (có sẵn; cần phát hiện bong bóng mây)
    "moan":      "MTO COMIC 2.ttf",          # rên / khoái cảm         (có sẵn)
    "sfx":       "Bangers-Regular.ttf",      # SFX tượng thanh         (dùng chung font impact với anger)
    "shout":     "MTO Damn Noisy Kids.ttf",  # thoại hét               (Việt hoá, đã tải)
    "narration": "Itim-Regular.ttf",         # thoại dẫn truyện        (Google Fonts, full tiếng Việt, chữ thường thật — thay MTO Chaney bị small-caps "Ta tU lUyệN")
    "anger":     "Bangers-Regular.ttf",      # tức giận / hiệu ứng     (Google Fonts, full tiếng Việt) — thay Badaboom BB (font Anh, ra ô vuông)
    "fear":      "ShantellSans.ttf",         # sợ hãi / run rẩy        (Google Fonts, full tiếng Việt, nét run) — thay Shiver Me Timbers (font Anh)
    # "horror":  "MTO Chiller.ttf",          # để dành cho cảnh kinh dị / special (chưa map kind nào)
}
# Hét: dấu '!' lặp (!!), hoặc '?!' / '!?' (kể cả fullwidth). KHÔNG khớp 1 dấu '!'
# đơn lẻ vì thoại thường tiếng Việt cũng hay kết thúc bằng '!'.
_SHOUT_RE = re.compile(r"[!！]{2,}|[?？][!！]|[!！][?？]")


def _is_manual_run() -> bool:
    """True khi cả run này là PASS VẼ TAY (✏️ vùng chữ thủ công).

    Pass thủ công chạy subprocess RIÊNG với detector=none và env MIT_MANUAL_REGIONS
    được set (xem _mit_backend._mit_manual_pass) → MỌI region trong run này đều là
    box người dùng vẽ. Dùng làm NGOẠI LỆ: vùng thủ công được phép nới ngang+dọc
    kể cả khi trông như bong bóng (người dùng chủ động đặt chỗ, chấp nhận tràn)."""
    return bool(os.environ.get("MIT_MANUAL_REGIONS", "").strip())


def _resolve_font_paths(primary_font: str) -> dict:
    """Trả dict kind→đường-dẫn-font. File không tồn tại → fallback primary_font."""
    base_dir = Path(primary_font).parent if primary_font else None
    resolved = {}
    for kind, fname in _FONT_BY_KIND.items():
        candidate = (base_dir / fname) if base_dir else None
        resolved[kind] = str(candidate) if (candidate and candidate.exists()) else primary_font
    return resolved


def _region_is_bubble(region) -> bool:
    """Heuristic: chữ nằm trong bong bóng (nền sáng / tương phản cao) hay trên artwork."""
    try:
        fg_col, bg_col = region.get_font_colors()
        return float(np.mean(bg_col)) > 140 or color_difference(fg_col, bg_col) > 100
    except Exception:
        return True  # mặc định coi là bong bóng (an toàn)


def _llm_type_for(region):
    """Loại thoại do LLM gán khi dịch (lưu ở manga_translator._VI_REGION_TYPES, khoá
    theo text gốc). Trả None nếu không có."""
    try:
        import manga_translator as _mt
        d = getattr(_mt, "_VI_REGION_TYPES", None)
        if not d:
            return None
        key = (getattr(region, "text", "") or "").strip()
        return d.get(key)
    except Exception:
        return None


def _classify_region(region) -> str:
    """Phân loại loại thoại để chọn font (HYBRID: LLM + hình dạng bong bóng).
    - LLM (đọc nội dung) đáng tin cho: rên (moan), hét (shout), dẫn truyện, sfx.
    - Hình dạng bong bóng đáng tin cho: suy nghĩ (mây) — vì nói↔nghĩ LLM hay nhầm.
    """
    text = (region.get_translation_for_rendering() or "").strip()
    llm = _llm_type_for(region)
    kind = getattr(region, "_bubble_kind", None)  # 'oval'|'cloud'|'rect' (bubble-fit gắn)

    # Sợ hãi / run rẩy: tin LLM hoàn toàn (theo nội dung) → font Shantell Sans (nét run).
    if llm == "fear":
        return "fear"
    # Tức giận: tin LLM → font Bangers (impact). Đặt TRƯỚC shout vì "tức giận" ≠ "gào hét"
    # (LLM phân biệt) — dù câu có '!!' vẫn ưu tiên nhãn anger của LLM.
    if llm == "anger":
        return "anger"
    # Hét: dấu '!!'/'?!' trong chữ, HOẶC LLM gán shout, HOẶC burst gai-nền-đen.
    # all_caps_short chỉ là PHỎNG ĐOÁN kiểu chữ (viết HOA, ngắn) → (a) cần ≥2 chữ cái
    # để 1 ký tự SFX lẻ như "A"/"W" KHÔNG bị tưởng hét, (b) KHÔNG đè khi LLM đã gán
    # loại nội dung khác (sfx/moan/thought/narration) — chỉ áp khi LLM bỏ trống/speech.
    letters = [c for c in text if c.isalpha()]
    all_caps_short = (len(letters) >= 2 and len(text) <= 14 and all(c.isupper() for c in letters)
                      and llm in (None, "", "speech", "shout"))
    if _SHOUT_RE.search(text) or all_caps_short or llm == "shout" or kind == "burst":
        return "shout"
    # Rên / khoái cảm: tin LLM → font Comic 2 (mềm, viết tay).
    if llm == "moan":
        return "moan"
    # Tượng thanh khác: tin LLM → font Bangers (impact, dùng chung anger).
    if llm == "sfx":
        return "sfx"
    # Dẫn truyện: khung chữ nhật HOẶC LLM nói narration.
    if kind == "rect" or llm == "narration":
        return "narration"
    # Suy nghĩ: hình mây HOẶC LLM gán 'thought' (mà KHÔNG phải khung dẫn/burst hét).
    # Trước đây CHỈ tin hình mây vì sợ LLM nhầm nói→nghĩ; nhưng nhiều manga vẽ suy nghĩ
    # trong bóng oval/không viền (không phải mây) → cloud-detect bắt trượt, font suy nghĩ
    # bị áp thiếu. Nay tin thêm nhãn LLM khi shape không phải rect (dẫn truyện) / burst
    # (hét) — hai loại này đã được xử lý ở các nhánh trên nên guard chỉ để phòng đổi thứ tự.
    if kind == "cloud" or (llm == "thought" and kind not in ("rect", "burst")):
        return "thought"
    # Chữ ngoài bong bóng (trên artwork): TRƯỚC ĐÂY ép hết thành "sfx" (font Bangers
    # impact) → câu thoại dài đặt trên nền tranh bị méo (vd "Ta là phàm tu, muốn làm
    # ma." ra font đập). Nay:
    #   - LLM đã khẳng định "speech" → tin LLM, giữ font thường dù nằm trên artwork.
    #   - LLM bỏ trống → chỉ chữ NGẮN (tượng thanh thật: "Bùm", "Rầm", "hà hà hà")
    #     mới dùng font SFX; câu dài coi như thoại thường.
    if not _region_is_bubble(region):
        if llm == "speech":
            return "normal"
        if llm in (None, "") and len(text) <= 14:
            return "sfx"
    return "normal"


def _bubble_interior_rect(img: np.ndarray, region) -> list | None:
    """Dò ruột bong bóng chứa vùng text → trả [x1,y1,x2,y2] (toạ độ ảnh) hoặc None.

    Ý tưởng (lưu ý #1): sau inpaint, ruột bong bóng sạch & đồng màu. floodFill từ
    TÂM text trên ảnh xám sẽ lan đến viền bong bóng tối → bbox vùng lan = bong bóng.
    Dùng để đặt chữ TO & CĂN GIỮA theo bong bóng thật, không phụ thuộc unclip.
    Trả None (→ fallback hành vi cũ) khi: lan tràn gần kín cửa sổ (nền CG/không kín),
    vùng quá nhỏ/lớn, hoặc không bao được text. Mọi lỗi đều nuốt → None."""
    try:
        x1, y1, x2, y2 = [int(v) for v in region.xyxy]
    except Exception:
        return None
    H, W = img.shape[:2]
    tw, th = x2 - x1, y2 - y1
    if tw < 4 or th < 4:
        return None
    # Cửa sổ quanh text (nới ~1.1× mỗi bên) — đủ bao trọn bong bóng thường gặp.
    ex, ey = int(tw * 1.1), int(th * 1.1)
    wx1, wy1 = max(0, x1 - ex), max(0, y1 - ey)
    wx2, wy2 = min(W, x2 + ex), min(H, y2 + ey)
    crop = img[wy1:wy2, wx1:wx2]
    if crop.size == 0:
        return None
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        ch, cw = gray.shape[:2]
        sx = max(0, min(cw - 1, int((x1 + x2) / 2) - wx1))
        sy = max(0, min(ch - 1, int((y1 + y2) / 2) - wy1))
        ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
        cv2.floodFill(gray.copy(), ff_mask, (sx, sy), 255,
                      loDiff=18, upDiff=18,
                      flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        filled = ff_mask[1:-1, 1:-1]
    except Exception:
        return None
    ys, xs = np.where(filled > 0)
    if xs.size < 64:
        return None
    bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = bx2 - bx1, by2 - by1
    fill = float(xs.size) / float(cw * ch)
    # Lan kín gần hết cửa sổ (không phải bong bóng kín) / quá nhỏ / quá to → bỏ.
    if (bw >= cw * 0.97 and bh >= ch * 0.97) or fill > 0.9 or fill < 0.06:
        return None
    ix1, iy1, ix2, iy2 = bx1 + wx1, by1 + wy1, bx2 + wx1, by2 + wy1
    # Phải bao được text và không phình quá mức (tránh bắt nhầm cả panel).
    if (ix2 - ix1) < tw * 0.9 or (iy2 - iy1) < th * 0.9:
        return None
    if (ix2 - ix1) * (iy2 - iy1) > tw * th * 12:
        return None
    # Bóng to bất thường so với CẢ TRANG → không phải bong bóng thoại. Trang nền
    # đen toàn chữ (trang dẫn truyện): flood phủ trọn nền đen rồi dừng ở LỀ TRẮNG
    # mép trang — biên có tương phản thật nên ring-check bên dưới không bắt được,
    # phải chặn bằng kích thước. Bong bóng thật hiếm khi vượt 1/3 trang hay trải
    # dài 3/4 bề cao/bề rộng trang (slab nền đen thường cao gần trọn trang).
    if (ix2 - ix1) * (iy2 - iy1) > 0.35 * (W * H):
        return None
    if (ix2 - ix1) > 0.75 * W or (iy2 - iy1) > 0.75 * H:
        return None

    # ── Chặn BÓNG ẢO: biên flood phải là VIỀN MỰC thật ─────────────────────────
    # Trang nền đen/nền phẳng (vd trang dẫn truyện toàn chữ): flood dừng ở nhiễu
    # JPEG chứ không phải viền bóng → rect bắt được là NGẪU NHIÊN → bubble-fit bóp
    # chữ Việt thành tháp hẹp + neo sai chỗ. Viền bóng thật luôn TƯƠNG PHẢN rõ với
    # ruột (nét mực tối quanh ruột trắng / viền sáng quanh burst đen); nhiễu thì
    # vành ngoài ≈ ruột. Vành = dilate(filled) − filled (2-3px sát biên flood).
    try:
        _ring = cv2.dilate(filled, np.ones((3, 3), np.uint8), iterations=2)
        _ring = (_ring > 0) & (filled == 0)
        if int(_ring.sum()) >= 32:
            _inner_mean = float(gray[filled > 0].mean())
            _ring_mean = float(gray[_ring].mean())
            if abs(_inner_mean - _ring_mean) < 18.0:
                return None
    except Exception:
        pass

    # ── Phân loại HÌNH DẠNG bong bóng (để chọn font) ──────────────────────────
    # rect (khung dẫn truyện): diện-tích/bbox cao (~0.9+); oval ~0.78.
    # cloud (bong bóng mây): viền gồ ghề → độ tròn 4π·S/P² thấp. Ngưỡng CHẶT +
    # chặn nhầm oval thon (aspect, rect_ratio) để không gán nhầm thoại thường.
    try:
        area = float(xs.size)
        rect_ratio = area / float(max(1, bw * bh))
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = max((cv2.arcLength(c, True) for c in cnts), default=0.0)
        circ = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 1.0
        aspect = bw / float(max(1, bh))
        inner_bright = float(gray[filled > 0].mean()) if xs.size else 255.0
        if rect_ratio > 0.90:
            region._bubble_kind = 'rect'
        elif circ < 0.55 and 0.55 <= rect_ratio <= 0.86 and 0.55 <= aspect <= 1.8:
            # Gai + độ tròn thấp: ruột TỐI (chữ trắng/nền đen) = burst NHẤN MẠNH/HÉT;
            # ruột SÁNG (chữ đen/nền trắng) = bong bóng mây SUY NGHĨ.
            region._bubble_kind = 'burst' if inner_bright < 110 else 'cloud'
        else:
            region._bubble_kind = 'oval'
    except Exception:
        region._bubble_kind = 'oval'
    return [ix1, iy1, ix2, iy2]


def _bubble_border_rect(img: np.ndarray, region) -> list | None:
    """Dò bong bóng bằng ĐƯỜNG VIỀN (Canny + contour) → [x1,y1,x2,y2] hoặc None.

    Dùng làm FALLBACK khi floodfill (ruột) thất bại — vd bong bóng MỜ / bán trong suốt
    đè lên nền: ruột KHÔNG đồng màu nên floodfill lan tràn, nhưng VIỀN bóng vẫn còn.
    Tìm contour KÍN, ĐẶC (lồi), BAO TRỌN text, NHỎ NHẤT và KHÔNG phải cả panel.
    Mọi lỗi / không chắc → None (an toàn: rơi về free-scale)."""
    try:
        x1, y1, x2, y2 = [int(v) for v in region.xyxy]
    except Exception:
        return None
    H, W = img.shape[:2]
    tw, th = x2 - x1, y2 - y1
    if tw < 4 or th < 4:
        return None
    # Cửa sổ rộng (bóng thường to hơn text ~1.5× mỗi bên) — có trần để khỏi ôm cả panel.
    ex, ey = int(tw * 1.5), int(th * 1.5)
    wx1, wy1 = max(0, x1 - ex), max(0, y1 - ey)
    wx2, wy2 = min(W, x2 + ex), min(H, y2 + ey)
    crop = img[wy1:wy2, wx1:wx2]
    if crop.size == 0:
        return None
    cw_, chh = wx2 - wx1, wy2 - wy1
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        med = float(np.median(gray))
        lo = int(max(0, 0.5 * med)); hi = int(min(255, 1.3 * med))
        edges = cv2.Canny(gray, lo, hi)
        # Khép HỞ viền mờ/đứt nét bằng morphological close → contour kín.
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kk, iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return None
    if not cnts:
        return None
    # Toạ độ text trong crop.
    ctx1, cty1, ctx2, cty2 = x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1
    tcx, tcy = (ctx1 + ctx2) / 2.0, (cty1 + cty2) / 2.0
    text_area = float(max(1, tw * th))
    win_area = float(max(1, cw_ * chh))
    slack_x, slack_y = tw * 0.12, th * 0.12
    best = None  # (area, (bx,by,bw,bh), contour)
    for c in cnts:
        bx, by, bbw, bbh = cv2.boundingRect(c)
        area = bbw * bbh
        # Phải BAO TRỌN text (lề nhỏ) và không phải gần cả cửa sổ.
        if not (bx <= ctx1 + slack_x and by <= cty1 + slack_y and
                bx + bbw >= ctx2 - slack_x and by + bbh >= cty2 - slack_y):
            continue
        if area < text_area * 0.8 or area > win_area * 0.92:
            continue
        # Tâm text phải NẰM TRONG contour (kín thật, không chỉ chồng bbox).
        if cv2.pointPolygonTest(c, (float(tcx), float(tcy)), False) < 0:
            continue
        # ĐẶC/LỒI (bóng tròn/oval/chữ nhật) → loại viền artwork lằng nhằng.
        cnt_area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (cnt_area / hull_area) < 0.80:
            continue
        # Chọn contour bao text NHỎ NHẤT (bóng sát text, không ôm panel ngoài).
        if best is None or area < best[0]:
            best = (area, (bx, by, bbw, bbh), c)
    if best is None:
        return None
    _, (bx, by, bbw, bbh), c = best
    # Trần kích thước theo TRANG (đồng bộ _bubble_interior_rect): bóng thật không
    # chiếm quá 1/3 trang hay trải 3/4 bề cao/bề rộng trang.
    if bbw * bbh > 0.35 * (W * H) or bbw > 0.75 * W or bbh > 0.75 * H:
        return None
    # Viền phải là NÉT MỰC thật: ruột contour vs pixel TRÊN đường viền phải tương
    # phản rõ. Cạnh GIẢ (mép blotch inpaint / vệt nén trên nền đen) chênh chỉ vài
    # mức xám → loại, kẻo chữ bị bubble-fit vào khung ngẫu nhiên.
    try:
        m_in = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m_in, [c], -1, 255, -1)
        m_line = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m_line, [c], -1, 255, 3)
        m_inner = (m_in > 0) & (m_line == 0)
        if int(m_inner.sum()) >= 64 and int((m_line > 0).sum()) >= 32:
            if abs(float(gray[m_inner].mean()) - float(gray[m_line > 0].mean())) < 18.0:
                return None
    except Exception:
        pass
    ix1, iy1, ix2, iy2 = bx + wx1, by + wy1, bx + bbw + wx1, by + bbh + wy1

    # Phân loại HÌNH DẠNG (giống _bubble_interior_rect, dùng để chọn font).
    try:
        cnt_area = float(cv2.contourArea(c))
        rect_ratio = cnt_area / float(max(1, bbw * bbh))
        perim = cv2.arcLength(c, True)
        circ = (4.0 * np.pi * cnt_area / (perim * perim)) if perim > 0 else 1.0
        aspect = bbw / float(max(1, bbh))
        if rect_ratio > 0.90:
            region._bubble_kind = 'rect'
        elif circ < 0.55 and 0.55 <= rect_ratio <= 0.86 and 0.55 <= aspect <= 1.8:
            region._bubble_kind = 'cloud'
        else:
            region._bubble_kind = 'oval'
    except Exception:
        region._bubble_kind = 'oval'
    return [ix1, iy1, ix2, iy2]


def _centered_text_block(fs: int, translation: str, bub_in, lang: str, img_w: int, img_h: int):
    """Trả dst_points (1,4,2) = khối chữ ở cỡ `fs`, căn GIỮA vùng bub_in (không lấp đầy).
    Tách riêng để tính lại được sau khi cap cỡ chữ (cap mới có tác dụng thật)."""
    ix1, iy1, ix2, iy2, cx, cy = bub_in
    bw = max(8, ix2 - ix1)
    bh = max(8, iy2 - iy1)
    try:
        lns, wds = text_render.calc_horizontal(fs, translation, max_width=bw, max_height=bh, language=lang)
    except Exception:
        lns, wds = [translation], None
    n = max(1, len(lns))
    tw = int(max(wds)) if wds else bw
    lsf = text_render._line_spacing_frac()  # khớp spacing_y ở put_text_horizontal
    th = int(n * fs + (n - 1) * fs * lsf)
    pad = int(fs * 0.5)
    box_w = max(8, min(bw, tw + pad))
    box_h = max(8, min(bh, th + pad))
    nx1 = int(max(ix1, min(cx - box_w // 2, ix2 - box_w)))
    ny1 = int(max(iy1, min(cy - box_h // 2, iy2 - box_h)))
    dp = np.array([[[nx1, ny1], [nx1 + box_w, ny1],
                    [nx1 + box_w, ny1 + box_h], [nx1, ny1 + box_h]]], dtype=np.int64)
    dp[..., 0] = dp[..., 0].clip(0, img_w - 1)
    dp[..., 1] = dp[..., 1].clip(0, img_h - 1)
    return dp

def _narrow_width_mult() -> float:
    """Hệ số NỚI BỀ NGANG cho vùng chữ DỌC HẸP (tuỳ chọn UI → env MIT_NARROW_WIDTH_MULT).
    Rỗng / < 1.0 = TẮT (giữ hành vi cũ, free-scale giữ tỉ lệ). Vd 2.0 = cho phép khối
    chữ Việt rộng gấp ~2× cột gốc; trần 6.0 để khỏi tràn cả trang."""
    try:
        v = float(os.environ.get("MIT_NARROW_WIDTH_MULT", "") or 0)
    except (TypeError, ValueError):
        return 1.0
    if v < 1.0:
        return 1.0
    return min(v, 6.0)


def _narrow_font_cap() -> float:
    """TRẦN CỠ CHỮ (px) cho vùng dọc hẹp ĐÃ nới ngang (tuỳ chọn UI → env
    MIT_NARROW_FONT_CAP). Rỗng / ≤ 0 = TẮT (giữ cỡ tự nhiên target_font_size).
    Khi đặt (vd 48), chữ ở khối nới-ngang bị ghìm ≤ trần → khối ngang nhưng chữ
    NHỎ lại (wrap nhiều dòng hơn), tránh chữ to choán cả panel."""
    try:
        v = float(os.environ.get("MIT_NARROW_FONT_CAP", "") or 0)
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 else 0.0


# Bề cao / bề rộng CỦA CHÍNH BOX ≥ ngưỡng này → coi là "hẹp chiều rộng" (cột dọc).
_NARROW_RATIO = 4.0


def _is_narrow_region(region) -> bool:
    """Box vùng chữ có HẸP CHIỀU RỘNG không: SO bề rộng với bề cao CỦA CHÍNH NÓ
    (bề cao / bề rộng ≥ _NARROW_RATIO). Quyết định có nới ngang hay không."""
    try:
        poly0 = Polygon(region.unrotated_min_rect[0])
        minx, miny, maxx, maxy = poly0.bounds
        ow = max(1.0, maxx - minx)
        oh = max(1.0, maxy - miny)
        return (oh / ow) >= _NARROW_RATIO
    except Exception:
        return False


def _scaled_box_for_floor(region, font_size: int, img: np.ndarray):
    """FREE SCALE GIỮ TỈ LỆ: phóng to Ô GỐC (min_rect) ĐỀU CẢ 2 CHIỀU (xfact=yfact=k)
    đến khi text vừa ở `font_size`. GIỮ NGUYÊN hướng + tỉ lệ ô gốc → ô 10×20 thành
    10k×20k, KHÔNG đổi chữ dọc thành chữ ngang (KHÔNG reflow). render() warp chữ lấp
    đầy ô đã đủ to nên chữ ≈ font_size (≥ min), không bị co. Trả dst_points (1,4,2)
    hoặc None nếu lỗi.

    NGOẠI LỆ — nới ngang vùng dọc hẹp (MIT_NARROW_WIDTH_MULT > 1): cột chữ dọc CJK rất
    hẹp khiến câu Việt bị nhồi thành cột mảnh/chữ tí. Khi bật, vùng cao-hẹp (oh ≥ 1.3×ow)
    được ÉP RENDER NGANG và cho bề rộng nới tới mult× cột gốc → chữ Việt chạy thành
    KHỐI NGANG rộng, dễ đọc. Cao của khối = vừa đủ chứa text wrap ở font_size."""
    try:
        poly0 = Polygon(region.unrotated_min_rect[0])
        minx, miny, maxx, maxy = poly0.bounds
        ow = max(1.0, maxx - minx)
        oh = max(1.0, maxy - miny)
        lang = getattr(region, "target_lang", "en_US")
        line_h = font_size * (1.0 + text_render._line_spacing_frac())  # khớp spacing_y ở put_text_horizontal
        Wimg, Himg = img.shape[1], img.shape[0]

        # ── LÀN CỘT (trang free-text CJK dọc, đọc trên→dưới phải→trái) ────────
        # Khối Việt giữ NGUYÊN làn cột gốc: rộng ≈ bề rộng cột (sàn ~3.2 cỡ chữ
        # để chứa được từ dài), NEO ĐỈNH cột + tâm-x cột → các cột giữ đúng vị
        # trí và thứ tự phải→trái như bản gốc, không phình thành khối ngang to
        # đè cột bên cạnh (đè nhau → _separate_blocks xáo hết vị trí).
        if getattr(region, '_col_keep', False):
            lane = min(0.45 * Wimg, max(ow, font_size * 3.2))
            lines, widths = text_render.calc_horizontal(
                font_size, region.translation,
                max_width=int(lane), max_height=int(0.92 * Himg), language=lang)
            n = max(1, len(lines))
            bw = max(8.0, min(lane, float(max(widths)) if widths else lane))
            bh = max(8.0, min(0.92 * Himg, n * line_h + line_h * 0.14))
            poly = affinity.scale(poly0, xfact=bw / ow, yfact=bh / oh,
                                  origin=(minx + ow / 2.0, miny))
            logger.info(f'[col-keep] "{region.get_translation_for_rendering()[:18]}" '
                        f'cột {int(ow)}×{int(oh)} → làn {int(bw)}×{int(bh)} '
                        f'neo đỉnh, {n} dòng')
            return _finalize_scaled_poly(poly, region, img, Wimg, Himg)

        # ── Nới ngang vùng DỌC HẸP (tuỳ chọn) ────────────────────────────────
        # Box HẸP CHIỀU RỘNG (bề cao ≥ _NARROW_RATIO × bề rộng — xem _is_narrow_region).
        # Box ngang/vuông không bao giờ lọt. Không phụ thuộc region.horizontal để vòng
        # recompute sau cap (gọi lại hàm này) luôn nhất quán dù _direction đã đổi.
        wmult = _narrow_width_mult()
        widen = wmult > 1.0 and _is_narrow_region(region)
        if widen:
            # Ép render NGANG (render() đọc region._direction): chữ Việt chạy ngang.
            try:
                region._direction = 'horizontal'
            except Exception:
                pass
            # GHÌM CỠ CHỮ (tuỳ chọn): trần px cho khối nới-ngang. Cap nhỏ → target_w
            # (= …font_size*6) và bh đều co theo → khối ngang nhưng chữ NHỎ lại, wrap
            # nhiều dòng hơn. Không đặt = giữ cỡ tự nhiên như trước.
            fcap = _narrow_font_cap()
            if fcap > 0 and font_size > fcap:
                font_size = int(fcap)
                line_h = font_size * (1.0 + text_render._line_spacing_frac())
            # Bề rộng đích: nới mult× cột gốc, tối thiểu ~6 chữ, trần 0.9 bề rộng ảnh.
            target_w = min(0.90 * Wimg, max(ow * wmult, font_size * 6.0))
            lines, widths = text_render.calc_horizontal(
                font_size, region.translation,
                max_width=int(target_w), max_height=int(0.90 * Himg), language=lang)
            n = max(1, len(lines))
            bw = max(8.0, min(target_w, float(max(widths)) if widths else target_w))
            bh = max(8.0, min(0.90 * Himg, n * line_h + line_h * 0.14))
            xfact = bw / ow
            yfact = bh / oh
            logger.info(f'[narrow-wide] "{region.get_translation_for_rendering()[:18]}" '
                        f'cột {int(ow)}×{int(oh)} → khối ngang ~{int(bw)}×{int(bh)} '
                        f'(×{wmult:g} width, {n} dòng)')
            poly = affinity.scale(poly0, xfact=xfact, yfact=yfact, origin='center')
            return _finalize_scaled_poly(poly, region, img, Wimg, Himg)

        # ── KHÔNG hẹp + vùng RỘNG (ngang) → CHỈ SCALE CHIỀU DÀI (cao) ─────────
        # Bong bóng KHÔNG dò được (bub_in=None) nhưng chữ vốn nằm trong 1 bong bóng:
        # GIỮ BỀ RỘNG GỐC (≈ bề rộng bong bóng do OCR cắt dòng) và cho chữ Việt WRAP
        # XUỐNG NHIỀU DÒNG (cao tăng) → lấp đúng khung, KHÔNG kéo dài thành banner
        # full-trang như free-scale giữ tỉ lệ. Chỉ áp cho vùng ngang (ow ≥ oh) — vùng
        # cao-hẹp đã đi nhánh nới-ngang ở trên; vùng dọc ratio<3 vẫn free-scale bên dưới.
        if ow >= oh:
            keep_w = min(ow, 0.90 * Wimg)
            lines, widths = text_render.calc_horizontal(
                font_size, region.translation,
                max_width=int(keep_w), max_height=int(0.90 * Himg), language=lang)
            n = max(1, len(lines))
            bw = max(8.0, min(keep_w, float(max(widths)) if widths else keep_w))
            bh = max(8.0, min(0.90 * Himg, n * line_h + line_h * 0.14))
            logger.info(f'[len-only] "{region.get_translation_for_rendering()[:18]}" '
                        f'giữ rộng {int(ow)} → cao {int(bh)} ({n} dòng, no-bubble)')
            poly = affinity.scale(poly0, xfact=bw / ow, yfact=bh / oh, origin='center')
            return _finalize_scaled_poly(poly, region, img, Wimg, Himg)

        # ── Vùng cao hơn rộng nhưng CHƯA đủ hẹp → tìm "làn" vừa chữ rồi dựng hộp ──
        # Vòng k chỉ để TÌM bề rộng làn (≤4× ô gốc, ≤0.9 ảnh) mà chữ wrap vừa ở
        # font_size — KHÔNG dùng ô-gốc-phóng-to làm hộp nữa.
        k_img = min((0.90 * Wimg) / ow, (0.90 * Himg) / oh)
        CAP = max(1.0, min(4.0, k_img))
        best_tw, best_th = None, None
        for k in (1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0):
            if k > CAP:
                break
            sw, sh = ow * k, oh * k
            if region.horizontal:
                # LÀN TỐI THIỂU cho chữ NGANG: cột CJK gốc rất hẹp (60-120px) làm
                # câu Việt wrap 1-2 từ/dòng thành "tháp" khó đọc. Cho làn rộng ít
                # nhất ~6 cỡ chữ (≈4-5 từ/dòng), trần 0.9 ảnh. Chữ DỌC giữ nguyên
                # (tháp là đúng hướng đọc dọc).
                sw = min(0.90 * Wimg, max(sw, font_size * 6.0))
                lines, widths = text_render.calc_horizontal(
                    font_size, region.translation,
                    max_width=int(sw), max_height=int(sh), language=lang)
                tw = max(widths) if widths else ow
                th = max(line_h, len(lines) * line_h)
            else:
                lines, heights = text_render.calc_vertical(
                    font_size, region.translation, max_height=int(sh))
                tw = len(lines) * line_h   # mỗi cột rộng ~font_size (+spacing)
                th = max(heights) if heights else 0
            best_tw, best_th = tw, th
            if tw <= sw and th <= sh:
                break

        # Hộp = KÍCH THƯỚC THẬT của khối chữ ở font_size (KHÔNG giữ ô gốc to):
        # render() warp chữ LẤP ĐẦY hộp — cột CJK gốc thường RỘNG/CAO hơn khối chữ
        # Việt tự nhiên, giữ nguyên ô gốc sẽ THỔI cỡ hiển thị vượt xa fs (vd fs=45
        # hiện ~79) → phá đồng bộ cỡ chữ toàn trang. Hộp đúng cỡ ⇒ hiển thị == fs.
        pad = font_size * 0.25
        bw = max(8.0, min(0.90 * Wimg, (best_tw or ow) + pad))
        bh = max(8.0, min(0.90 * Himg, (best_th or oh) + pad))
        poly = affinity.scale(poly0, xfact=bw / ow, yfact=bh / oh, origin='center')
        return _finalize_scaled_poly(poly, region, img, Wimg, Himg)
    except Exception:
        return None


def _finalize_scaled_poly(poly, region, img, Wimg, Himg):
    """Xoay poly (đã scale ở hệ chưa xoay) về toạ độ ảnh, DỜI NGUYÊN KHỐI vào trong
    ảnh (không squash méo) rồi clip an toàn. Tách riêng để cả nhánh free-scale lẫn
    nhánh nới-ngang dùng chung. Trả dst_points (1,4,2) hoặc None nếu lỗi."""
    try:
        pts = np.array(poly.exterior.coords[:4])
        dst = rotate_polygons(
            region.center, pts.reshape(1, -1), -region.angle, to_int=False,
        ).reshape(-1, 4, 2)
        # DỜI NGUYÊN KHỐI vào trong ảnh (không squash méo) rồi mới clip an toàn cuối.
        # Box đã ≤0.9 ảnh nên không thể chạm cả 2 mép → dời 1 phía là đủ.
        xs, ys = dst[..., 0], dst[..., 1]
        if xs.min() < 0:
            dst[..., 0] -= xs.min()
        elif xs.max() > Wimg - 1:
            dst[..., 0] -= (xs.max() - (Wimg - 1))
        if ys.min() < 0:
            dst[..., 1] -= ys.min()
        elif ys.max() > Himg - 1:
            dst[..., 1] -= (ys.max() - (Himg - 1))
        dst = dst.astype(np.int64)
        dst[..., 0] = dst[..., 0].clip(0, Wimg - 1)
        dst[..., 1] = dst[..., 1].clip(0, Himg - 1)
        return dst
    except Exception:
        return None


def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    # Extra: khi bg tối (vùng artwork không có speech bubble) mà fg cũng tối
    # → chữ ĐEN + stroke TRẮNG dày: outline trắng nổi bật trên nền tối (giống bản gốc tiếng Trung).
    bg_avg = np.mean(bg)
    if bg_avg < 100 and np.mean(fg) < 160:
        fg = (0, 0, 0)           # chữ đen
        bg = (255, 255, 255)     # stroke trắng dày bao quanh
    return fg, bg

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

# Ký tự VÔ HÌNH translator dùng làm placeholder (ZWJ cho watermark đã xoá, [cont]
# đã gộp…) — translation chỉ gồm những ký tự này nghĩa là "không render gì".
# (ZWSP, ZWNJ, ZWJ, WORD JOINER, BOM — viết dạng code-point vì literal không nhìn thấy.)
_INVISIBLE_TRANS = {cp: None for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)}


def _meaningful_translation(t) -> bool:
    """True nếu bản dịch có nội dung THẬT để render. Lọc trước khi tính cỡ chữ —
    placeholder ZWJ mà lọt qua sẽ được fit cỡ chữ rất to (1 ký tự / box rộng) rồi
    kéo lệch median của bước đồng bộ cỡ chữ toàn trang + tốn lượt render thừa."""
    if not t:
        return False
    try:
        return bool(str(t).translate(_INVISIBLE_TRANS).strip())
    except Exception:
        return True


def _merge_cont_regions(text_regions):
    """UNION vùng [cont] vào vùng anchor của nó.

    Một câu nguồn bị detector cắt thành nhiều vùng → LLM dịch TRỌN câu vào segment
    đầu (anchor) và đánh dấu các segment sau là [cont] (rule 6c trong system prompt).
    Translator ghi map (text cont → text anchor) ở manga_translator._VI_REGION_MERGES.
    Ở đây: nối lines của vùng cont vào anchor + xoá cache hình học (cached_property)
    → box anchor phình ra ôm đủ chỗ câu gộp; vùng cont bị xoá translation để bước
    lọc phía sau loại khỏi render. Không có map / lỗi → giữ nguyên (an toàn)."""
    try:
        import manga_translator as _mt
        merges = getattr(_mt, "_VI_REGION_MERGES", None)
    except Exception:
        return
    if not merges:
        return
    by_text = {}
    for r in text_regions:
        key = (getattr(r, 'text', '') or '').strip()
        if key and key not in by_text:
            by_text[key] = r
    geom_cache = ('xyxy', 'xywh', 'center', 'unrotated_polygons', 'unrotated_min_rect',
                  'min_rect', 'polygon_aspect_ratio', 'unrotated_size', 'aspect_ratio')
    for cont_key, anchor_key in merges.items():
        cont = by_text.get(cont_key)
        anchor = by_text.get(anchor_key)
        if cont is None or anchor is None or cont is anchor:
            continue
        try:
            # CHỈ union hình học khi 2 vùng THẬT SỰ KỀ NHAU (câu bị detector cắt
            # giữa chừng). Cặp merge nằm XA nhau (vd 2 nửa câu ở 2 panel khác nhau)
            # mà union thì box anchor phình thành dải dài xuyên panel → khối chữ
            # rơi đúng GIỮA dải, đè lên artwork/nhân vật. Khi xa: GIỮ box anchor,
            # câu gộp render tại CHỖ anchor; chỗ cont đã inpaint sạch, bỏ trống.
            ax1, ay1, ax2, ay2 = [float(v) for v in anchor.xyxy]
            cx1, cy1, cx2, cy2 = [float(v) for v in cont.xyxy]
            gap_x = max(0.0, max(ax1, cx1) - min(ax2, cx2))
            gap_y = max(0.0, max(ay1, cy1) - min(ay2, cy2))
            # Ngưỡng "kề": 1.5× cạnh NGẮN của box nhỏ hơn (≈ bề dày dòng/cột chữ,
            # tỉ lệ theo cỡ chữ thật), sàn 30px. Cột/cạnh dòng kề nhau luôn lọt;
            # 2 nhãn ở 2 panel cách cả nghìn px thì không.
            near_thr = max(30.0, 1.5 * min(
                min(ax2 - ax1, ay2 - ay1), min(cx2 - cx1, cy2 - cy1)))
            if max(gap_x, gap_y) > near_thr:
                # Vẫn nối texts để expansion_ratio tính trên câu nguồn ĐẦY ĐỦ.
                anchor.texts = list(anchor.texts) + list(cont.texts)
                cont.translation = ''
                logger.info(f'[merge-far] "{cont_key[:14]}…" cách "{anchor_key[:14]}…" '
                            f'{int(max(gap_x, gap_y))}px (> ngưỡng {int(near_thr)}) — '
                            f'render câu gộp tại vùng anchor, KHÔNG union box.')
                continue
            anchor.lines = np.concatenate(
                [np.asarray(anchor.lines), np.asarray(cont.lines)], axis=0)
            # Nối cả texts: used_rows / expansion_ratio tính theo câu nguồn ĐẦY ĐỦ.
            anchor.texts = list(anchor.texts) + list(cont.texts)
            for prop in geom_cache:
                anchor.__dict__.pop(prop, None)
            cont.translation = ''
            logger.info(f'[merge-cont] union vùng "{cont_key[:14]}…" vào "{anchor_key[:14]}…" '
                        f'(một câu bị detector cắt đôi).')
        except Exception:
            continue


def _bubble_max_fit(region, bub_in, lang: str, fs_start: int) -> int:
    """Cỡ chữ LỚN NHẤT ≤ fs_start mà bản dịch (wrap nhiều dòng) vừa ruột bóng bub_in.
    Dùng chung cho bubble-fit trong vòng chính lẫn bước đồng bộ cỡ chữ toàn trang."""
    ix1, iy1, ix2, iy2 = bub_in[:4]
    bw = max(8, ix2 - ix1)
    bh = max(8, iy2 - iy1)
    fs = max(1, int(fs_start))
    while fs > 1:
        lns, wds = text_render.calc_horizontal(
            fs, region.translation, max_width=bw, max_height=bh, language=lang)
        n = max(1, len(lns))
        tw = int(max(wds)) if wds else 0
        th = int(n * fs + (n - 1) * fs * text_render._line_spacing_frac())
        if tw <= bw and th <= bh:
            break
        fs -= 1
    return max(1, fs)


def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):
    """
    Adjust font size to fit translated text within the original speech bubble bbox.

    Strategy (Vietnamese / Latin text is ~1.3-1.5x longer than CJK source):
      1. Apply font_size_offset to get target_font_size.
      2. For horizontal regions: iteratively reduce font_size until the text wraps
         into <= used_rows rows inside the ORIGINAL bbox width/height.
         Only if even font_size_minimum cannot fit, allow a HEIGHT-only expansion
         (never WIDTH — width expansion overflows into adjacent artwork).
      3. General fallback: use original min_rect unchanged (no bbox scaling).
    """
    if font_size_minimum == -1:
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    font_size_minimum = max(1, font_size_minimum)

    # Cả run này là vùng vẽ tay? (✏️ thủ công = NGOẠI LỆ, được nới như artwork.)
    is_manual = _is_manual_run()

    dst_points_list = []
    bubble_jobs = []  # (index, region, bub_in, lang) tính lại khối sau khi cap cỡ chữ
    for region in text_regions:

        original_region_font_size = region.font_size
        if original_region_font_size <= 0:
            original_region_font_size = font_size_minimum

        # ── Compute expansion ratio (Vietnamese vs Chinese source) ──
        cn_chars = max(1.0, sum(count_text_length(t) for t in region.texts))
        vn_chars = max(1.0, count_text_length(region.translation))
        expansion_ratio = vn_chars / cn_chars

        # ── Adaptive factor: scale down font proportionally to how much longer VN is ──
        # Làm dịu so với trước (0.93/0.85/0.76) để chữ VI không bị bé;
        # vòng fit-to-bbox bên dưới vẫn thu lại nếu tràn nên an toàn.
        if expansion_ratio <= 1.3:
            adaptive_factor = 1.0
        elif expansion_ratio <= 1.8:
            adaptive_factor = 0.97
        elif expansion_ratio <= 2.5:
            adaptive_factor = 0.92
        else:
            adaptive_factor = 0.86

        if font_size_fixed is not None:
            target_font_size = font_size_fixed
        else:
            base = int(original_region_font_size * adaptive_factor)
            target_font_size = max(font_size_minimum, base + font_size_offset)

        target_font_size = max(target_font_size, font_size_minimum, 1)

        # Trần TUYỆT ĐỐI theo ảnh + ĐỘ DÀI bản dịch: OCR đôi khi trả font_size ảo
        # rất lớn (tiêu đề trang trí/con dấu bị gộp vùng — vd fs=166×0.86=143 lọt
        # ngay dưới trần 1/12 cũ), trang chỉ 1 vùng thì font-normalize (cần ≥2 vùng)
        # không chạy → chữ Việt phủ nửa trang. Câu càng DÀI trần càng THẤP (cỡ chữ
        # hợp lý tỉ lệ nghịch với lượng chữ phải chứa): ≤36 ký tự giữ trần 1/12 cũ
        # (tiêu đề ngắn vẫn được to), ~50 ký tự → 1/17, clamp 1/30 để câu dẫn rất
        # dài không bị ép sát font_min (fit-loop/bubble-fit vẫn tự thu thêm sau).
        _tlen = count_text_length(region.translation or "")
        _den = max(12.0, min(30.0, _tlen / 3.0))
        _abs_cap = max(font_size_minimum, int(round(min(img.shape[0], img.shape[1]) / _den)))
        if target_font_size > _abs_cap:
            logger.info(f'[font-abs-cap] "{(region.get_translation_for_rendering() or "")[:18]}" '
                        f'fs {target_font_size} → {_abs_cap} (trần cạnh ngắn/{_den:.0f}, '
                        f'dịch {int(_tlen)} ký tự).')
            target_font_size = _abs_cap

        # ── Detect if inside a styled speech bubble ──
        # High contrast (white-on-black / black-on-white) OR light background
        try:
            fg_col, bg_col = region.get_font_colors()
            bg_avg = np.mean(bg_col)
            contrast = color_difference(fg_col, bg_col)
            is_bubble = bg_avg > 140 or contrast > 100
        except Exception:
            is_bubble = True  # safer default

        # CHAT BOX = floodfill (bub_in) dò được ruột bóng KÍN — đáng tin hơn màu nền
        # (is_bubble theo màu hay nhận NGƯỢC: chữ trắng/nền tối artwork bị coi là bóng,
        # bóng hồng nhạt lại bị coi là artwork). Quyết định:
        #   • bub_in dò RA   → CHAT BOX → fit trong bóng, KHÔNG nới (nhánh bubble-fit).
        #   • bub_in KHÔNG ra → chữ TỰ DO → FREE SCALE (khối chữ đúng tỉ lệ, căn tâm).
        #   • ✏️ vùng thủ công → bỏ qua dò bóng → free scale theo ý người dùng vẽ.
        # is_bubble (màu) chỉ còn dùng cho heuristic rows_capacity, KHÔNG quyết định nới.

        dst_points = None
        single_axis_expanded = False

        if region.horizontal:
            used_rows = max(1, len(region.texts))

            # ── Lưu ý #1: căn theo BONG BÓNG thật ──────────────────────────────
            # Nếu dò được ruột bong bóng, dùng nó làm vùng fit + đặt chữ (căn giữa,
            # chữ to tự nhiên) thay vì vùng text unclip (ôm sát → chữ bị co nhỏ).
            # LUÔN thử dò chat box (KHÔNG gate theo màu). Vùng thủ công bỏ qua → free
            # scale theo ý người dùng vẽ. Hai tầng:
            #   1) floodfill ruột — chính xác khi ruột bóng sạch/đồng màu.
            #   2) FALLBACK viền (Canny+contour) — bắt bóng MỜ/đè nền mà floodfill trượt.
            bubble_rect = None
            if not is_manual:
                bubble_rect = _bubble_interior_rect(img, region)
                if bubble_rect is None:
                    bubble_rect = _bubble_border_rect(img, region)
                    if bubble_rect is not None:
                        logger.info(f'[bubble-border] "{region.get_translation_for_rendering()[:18]}" '
                                    f'bắt bóng bằng VIỀN (floodfill trượt).')
            bub_in = None  # (in_x1, in_y1, in_x2, in_y2, cx, cy) vùng đặt chữ (đã giới hạn)
            bub_src = 'bóng'  # khung chứa lấy từ đâu — chỉ để log cho dễ soi
            if bubble_rect is not None:
                bx1, by1, bx2, by2 = bubble_rect
                # Chừa lề trong để chữ không chạm viền bong bóng (x:7%, y:9%).
                mx = int((bx2 - bx1) * 0.07); my = int((by2 - by1) * 0.09)
                ix1, iy1, ix2, iy2 = bx1 + mx, by1 + my, bx2 - mx, by2 - my
                # Chặn theo VÙNG TEXT GỐC: GIỮ bề NGANG ≈ gốc (chỉ nới nhẹ 1.3×), cho
                # giãn CHIỀU CAO rộng hơn (3×). Giữ chữ trong "làn dọc" của bản gốc →
                # chữ Việt dài sẽ xuống NHIỀU DÒNG (cột hẹp) thay vì phình ngang lấn
                # sang bong bóng/vùng kế bên. Vùng đặt = ruột bóng ∩ (gốc đã nới).
                rcx, rcy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
                try:
                    ox1, oy1, ox2, oy2 = [int(v) for v in region.xyxy]
                    rcx, rcy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                    hx = max(1, int((ox2 - ox1) * 0.65))   # bề ngang ≈ 1.3× gốc
                    hy = max(1, int((oy2 - oy1) * 1.5))    # chiều cao ≈ 3× gốc
                    ix1 = max(ix1, rcx - hx); iy1 = max(iy1, rcy - hy)
                    ix2 = min(ix2, rcx + hx); iy2 = min(iy2, rcy + hy)
                except Exception:
                    pass
                if ix2 - ix1 >= 12 and iy2 - iy1 >= 12:
                    bbox_w = ix2 - ix1
                    bbox_h = iy2 - iy1
                    # Luôn neo theo TÂM GỐC của vùng (đặt chữ ĐÚNG CHỖ chữ gốc, không
                    # dồn về tâm bóng) → giữ layout gốc, không tự gây đè nhau. Bong
                    # bóng chỉ dùng để GIỚI HẠN cỡ (chữ to vừa, không tràn).
                    cx = max(ix1, min(rcx, ix2))
                    cy = max(iy1, min(rcy, iy2))
                    bub_in = (ix1, iy1, ix2, iy2, cx, cy)
            if is_manual and bub_in is None:
                # ✏️ Vùng tay = CHAT BOX do người dùng vẽ. Coi CHÍNH box đó là khung
                # chứa cố định: FIT chữ BÊN TRONG (calc_horizontal wrap nhiều dòng +
                # vòng co font ở nhánh bubble-fit), KHÔNG free-scale phình ra ngoài
                # (gây tràn rộng, không xuống dòng, đè artwork — đúng lỗi đang gặp).
                try:
                    ox1, oy1, ox2, oy2 = [int(v) for v in region.xyxy]
                    mx = int((ox2 - ox1) * 0.05); my = int((oy2 - oy1) * 0.06)
                    ix1, iy1, ix2, iy2 = ox1 + mx, oy1 + my, ox2 - mx, oy2 - my
                    if ix2 - ix1 >= 12 and iy2 - iy1 >= 12:
                        bbox_w = ix2 - ix1
                        bbox_h = iy2 - iy1
                        cx = (ix1 + ix2) // 2
                        cy = (iy1 + iy2) // 2
                        bub_in = (ix1, iy1, ix2, iy2, cx, cy)
                        bub_src = 'box tay'
                except Exception:
                    pass
            if bub_in is None and not is_manual:
                # KHÔNG dò ra bóng (floodfill lẫn viền đều trượt — vd vệt mực splash
                # trên nền biển/trời) → coi CHÍNH BOX TEXT detector cắt là khung chứa
                # cố định: FIT chữ BÊN TRONG (giảm font nếu cần) như hành vi cũ,
                # KHÔNG free-scale phình ra ngoài đè artwork.
                try:
                    ox1, oy1, ox2, oy2 = [int(v) for v in region.xyxy]
                    ow_, oh_ = ox2 - ox1, oy2 - oy1
                    # CỘT cao-hẹp + câu DÀI (≥6 từ): box-fit xếp tháp 1-từ/dòng rất
                    # khó đọc (trang toàn chữ: tiêu đề/cột dẫn truyện dài — tháp
                    # 362x1772) → BỎ box-fit, để free-scale dựng KHỐI NGANG quanh tâm
                    # cột. Nhãn ngắn (≤5 từ, vd "Kết Đan sơ kỳ.") vẫn tháp — đẹp trên
                    # vệt mực splash.
                    _nwords = len((region.translation or '').split())
                    if oh_ >= 3 * ow_ and _nwords >= 6:
                        # Trang FREE-TEXT CJK dọc (đọc trên→dưới, PHẢI→TRÁI): giữ
                        # LÀN cột gốc — khối Việt hẹp, neo ĐỈNH cột, tâm-x giữ
                        # nguyên (xem nhánh _col_keep trong _scaled_box_for_floor)
                        # → giữ đúng thứ tự đọc phải→trái, không choán chỗ cột bên.
                        region._col_keep = True
                        logger.info(f'[box-fit-skip] "{(region.get_translation_for_rendering() or "")[:18]}" '
                                    f'cột {ow_}x{oh_} + {_nwords} từ → làn cột (col-keep)')
                    else:
                        if oh_ > ow_:
                            # Cột DỌC-HẸP (CJK dọc): chữ Việt chạy ngang bị bề rộng cột
                            # bóp nghẹt (fs rơi 27/20 trong khi body trang 52) → nới
                            # ngang 1.5× + cao 1.15× quanh tâm. Vệt mực/splash vốn loe
                            # rộng hơn cột text nên nới nhẹ vẫn nằm trên nền bóng. Box
                            # ngang (caption/thoại) giữ NGUYÊN — không cần nới.
                            ex = int(ow_ * 0.25); ey = int(oh_ * 0.075)
                            ox1 = max(0, ox1 - ex); ox2 = min(img.shape[1] - 1, ox2 + ex)
                            oy1 = max(0, oy1 - ey); oy2 = min(img.shape[0] - 1, oy2 + ey)
                        if ox2 - ox1 >= 12 and oy2 - oy1 >= 12:
                            bbox_w = ox2 - ox1
                            bbox_h = oy2 - oy1
                            bub_in = (ox1, oy1, ox2, oy2,
                                      (ox1 + ox2) // 2, (oy1 + oy2) // 2)
                            bub_src = 'box gốc'
                            # Đánh dấu khung chứa là BOX THẬT của detector (không phải
                            # bóng dò) — van cứu "bóng ảo" ở font-normalize bỏ qua các
                            # vùng này (xem _apply_body_target).
                            region._box_fit = True
                            # Cỡ TỰ NHIÊN (trước khi bị box bóp) — median toàn trang sẽ
                            # dùng cỡ này thay vì cỡ đã fit, kẻo vài cột hẹp kéo sập
                            # body target (median 52 → 27, bóng thoại bị ghìm nhỏ theo).
                            region._natural_fs = int(target_font_size)
                except Exception:
                    pass
            if bub_in is None:
                bbox_h = region.unrotated_size[1]
                bbox_w = region.unrotated_size[0]

            # ── Step 1: reduce font until text fits within original bbox ──
            # rows_capacity = how many rows the bbox can physically hold at
            # current font size (≈ font_size × 1.2 line height).
            # This allows Vietnamese text to use MORE rows than the original
            # Chinese source when the bubble has room — avoiding tiny fonts.
            fit_size = target_font_size
            line_text_list, _ = text_render.calc_horizontal(
                fit_size,
                region.translation,
                max_width=bbox_w,
                max_height=bbox_h,
                language=getattr(region, "target_lang", "en_US"),
            )
            needed_rows = len(line_text_list)

            if is_bubble:
                # Inside styled bubble: allow wrapping to fill bbox height naturally
                def rows_capacity(fs):
                    return max(used_rows, int(bbox_h / max(1, fs * 1.1)))
            else:
                # Overlay on artwork: allow generous rows to preserve adaptive font size
                # Allowing +3 extra rows prevents the font from shrinking to illegible sizes
                def rows_capacity(fs):
                    return max(used_rows + 3, int(bbox_h / max(1, fs * 1.1)))

            while needed_rows > rows_capacity(fit_size) and fit_size > font_size_minimum:
                fit_size = max(font_size_minimum, fit_size - 1)
                line_text_list, _ = text_render.calc_horizontal(
                    fit_size,
                    region.translation,
                    max_width=bbox_w,
                    max_height=bbox_h,
                    language=getattr(region, "target_lang", "en_US"),
                )
                needed_rows = len(line_text_list)

            target_font_size = fit_size
            single_axis_expanded = True  # skip general-scaling fallback

            # ── Step 2: if still can't fit → expand HEIGHT (never width) ──
            # Bỏ qua khi đã đặt theo bong bóng (dst_points tính riêng bên dưới).
            if bub_in is None and needed_rows > rows_capacity(fit_size):
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    region_width = maxx - minx
                    img_width = img.shape[1]
                    scale_y = min(needed_rows / used_rows, 2.0)
                    if region_width > img_width * 0.5:
                        # Wide region (CG/subtitle): grow upward, anchor at bottom
                        poly = affinity.scale(poly, xfact=1.0, yfact=scale_y, origin=(minx, maxy))
                    else:
                        # Narrow region (manga bubble): grow quanh TÂM (lưu ý #1) — giữ
                        # chữ ở giữa bong bóng thay vì neo đỉnh (neo đỉnh đẩy chữ xuống,
                        # gây lệch). Giãn đối xứng → chữ vừa to hơn vừa căn giữa.
                        poly = affinity.scale(poly, xfact=1.0, yfact=scale_y, origin='center')
                    pts = np.array(poly.exterior.coords[:4])
                    dst_points = rotate_polygons(
                        region.center, pts.reshape(1, -1), -region.angle,
                        to_int=False,
                    ).reshape(-1, 4, 2)
                    dst_points = dst_points.astype(np.int64)
                    dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)
                    dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)
                except Exception:
                    pass

            # ── Đặt KHỐI CHỮ căn giữa, cỡ TỰ NHIÊN theo target_font_size ──────────
            # render() warp chữ để LẤP ĐẦY dst_points → dựng dst_points = đúng khối
            # chữ ở target_font_size (đã ≥ font_size_minimum nhờ fit-loop) thì cỡ
            # HIỂN THỊ không nhỏ hơn min → "Font min" trở thành SÀN thật.
            #   • Trong bong bóng (bub_in): clamp theo ruột bóng — chữ to vừa, không
            #     phình ra đè bóng cạnh.
            #   • Trên artwork (bub_in=None): TRƯỚC ĐÂY để dst_points = bbox gốc nhỏ
            #     → render warp THU chữ về cỡ bbox gốc nên set min KHÔNG ăn. Nay neo
            #     theo TÂM GỐC với vùng cho phép rộng → khối = cỡ tự nhiên ở fs ≥ min
            #     (chấp nhận có thể tràn artwork — theo lựa chọn người dùng).
            lang = getattr(region, "target_lang", "en_US")
            if bub_in is not None and _narrow_width_mult() > 1.0 and _is_narrow_region(region):
                # Luật 3: CHAT BOX (bong bóng) HẸP chiều rộng (cao/rộng ≥ _NARROW_RATIO)
                # + bật nới ngang → BỎ bubble-fit, dùng nhánh nới-ngang (ép render ngang,
                # khối rộng). Bong bóng thoại thường RỘNG nên nhánh này hiếm khớp — chỉ
                # chạy khi thực sự có bong bóng cao-hẹp.
                dp = _scaled_box_for_floor(region, target_font_size, img)
                if dp is not None:
                    dst_points = dp
                    bubble_jobs.append((len(dst_points_list), region, None, lang))
                    single_axis_expanded = True
                    logger.info(f'[narrow-wide/bubble] "{region.get_translation_for_rendering()[:18]}" '
                                f'chat box hẹp → nới ngang')
            elif bub_in is not None:
                # Trong bong bóng: AUTO-FIT theo ruột bóng, BỎ QUA font_min. Bóng có
                # kích thước cố định — ép tới Font min sẽ tràn/đè viền. Tìm cỡ lớn nhất
                # ≤ target (cỡ tự nhiên) mà chữ vừa ruột bóng; cho phép GIẢM xuống dưới
                # min nếu bóng nhỏ. (Font min chỉ là SÀN cho chữ NGOÀI bóng/artwork.)
                try:
                    target_font_size = _bubble_max_fit(region, bub_in, lang, target_font_size)
                    dst_points = _centered_text_block(
                        target_font_size, region.translation, bub_in, lang,
                        img.shape[1], img.shape[0])
                    # bub_in != None ⇒ recompute bằng _centered_text_block sau cap.
                    bubble_jobs.append((len(dst_points_list), region, bub_in, lang))
                    logger.info(f'[bubble-fit] "{region.get_translation_for_rendering()[:18]}" '
                                f'fs={target_font_size} trong {bub_src} (auto-fit, bỏ min)')
                except Exception:
                    pass
            else:
                # bub_in is None ⇒ KHÔNG dò ra bong bóng kín ⇒ chữ TỰ DO ⇒ FREE SCALE
                # GIỮ TỈ LỆ: phóng đều ô gốc cả 2 chiều (giữ hướng dọc/ngang gốc).
                dp = _scaled_box_for_floor(region, target_font_size, img)
                if dp is not None:
                    dst_points = dp
                    # bub_in=None ⇒ recompute bằng scale-box sau cap cỡ chữ.
                    bubble_jobs.append((len(dst_points_list), region, None, lang))
                    logger.info(f'[fit] "{region.get_translation_for_rendering()[:18]}" '
                                f'fs={target_font_size} (free-scale giữ tỉ lệ, no bubble, '
                                f'is_bubble_color={is_bubble})')

        if region.vertical:
            # ── Sàn cỡ chữ cho vùng DỌC (lưu ý #1, đồng bộ vùng ngang/artwork) ──────
            # TRƯỚC: nhánh dọc chỉ nới bề ngang ≤2× theo số cột cần thêm rồi để render()
            # warp ÉP chữ cho lấp đầy → cỡ HIỂN THỊ do box quyết định, "Font min" KHÔNG
            # thành sàn thật (câu Việt dọc dài bị co nhỏ bất kể min=30).
            # NAY: dùng chung _scaled_box_for_floor (đã hỗ trợ vertical qua calc_vertical)
            # — phóng to ô GỐC (giữ hướng dọc & tỉ lệ) đến khi text vừa ở target_font_size
            # ≥ font_min (trần ×3). Box đủ to ⇒ warp không thu chữ < min ⇒ min là SÀN thật.
            # Vùng dọc không dò bong bóng → coi như chữ tự do, FREE SCALE giữ tỉ lệ.
            lang = getattr(region, "target_lang", "en_US")
            dp = _scaled_box_for_floor(region, target_font_size, img)
            if dp is not None:
                dst_points = dp
                # bub_in=None ⇒ recompute bằng scale-box sau khi cap cỡ chữ toàn trang.
                bubble_jobs.append((len(dst_points_list), region, None, lang))
                single_axis_expanded = True
                logger.info(f'[fit] "{region.get_translation_for_rendering()[:18]}" '
                            f'fs={target_font_size} dọc (×box)')
            else:
                # Fallback (lỗi/suy biến): hành vi cũ — nới bề ngang theo số cột cần thêm.
                used_cols = max(1, len(region.texts))
                line_text_list, _ = text_render.calc_vertical(
                    region.font_size,
                    region.translation,
                    max_height=region.unrotated_size[1],
                )
                needed_cols = len(line_text_list)
                if needed_cols > used_cols:
                    scale_x = min(((needed_cols - used_cols) / used_cols) + 1, 2.0)
                    try:
                        poly = Polygon(region.unrotated_min_rect[0])
                        minx, miny, maxx, maxy = poly.bounds
                        poly = affinity.scale(poly, xfact=1.0, yfact=scale_x, origin=(minx, miny))
                        pts = np.array(poly.exterior.coords[:4])
                        dst_points = rotate_polygons(
                            region.center, pts.reshape(1, -1), -region.angle,
                            to_int=False,
                        ).reshape(-1, 4, 2)
                        dst_points = dst_points.astype(np.int64)
                        dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1] - 1)
                        dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0] - 1)
                        single_axis_expanded = True
                    except Exception:
                        pass

        # Ensure dst_points is never None — render() will crash if it is
        if dst_points is None:
            dst_points = region.min_rect

        # ── DIAG: vùng đi nhánh nào, box ra bao nhiêu (gỡ sau khi chỉnh xong) ──────
        try:
            _dp = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
            _bw = int(_dp[:, 0].max() - _dp[:, 0].min())
            _bh = int(_dp[:, 1].max() - _dp[:, 1].min())
            _fd = getattr(region, "_direction", None) or getattr(region, "direction", "?")
            logger.info(
                f'[size-diag] "{(region.get_translation_for_rendering() or "")[:18]}" '
                f'H={region.horizontal} V={region.vertical} dir={_fd} '
                f'is_bubble={locals().get("is_bubble")} bub_in={locals().get("bub_in") is not None} '
                f'fs={int(target_font_size)} min={font_size_minimum} box={_bw}x{_bh}'
            )
        except Exception:
            pass

        dst_points_list.append(dst_points)
        region.font_size = int(target_font_size)

    # ── Đồng bộ cỡ chữ BODY theo nhóm toàn trang (lưu ý 3: chữ to chữ nhỏ chênh) ──
    # Per-region auto-fit cho mỗi vùng MỘT cỡ tuỳ box dò được → cùng là thoại/dẫn
    # truyện mà chữ to chữ nhỏ lộn xộn. Chuẩn typesetting: MỘT cỡ body cho cả trang.
    #   • body (thoại/dẫn/suy nghĩ/rên/sợ): kéo VỀ CÙNG cỡ = median nhóm — vùng tự
    #     do dựng lại box theo cỡ mới (lên/xuống đều được), vùng trong bóng chỉ lên
    #     tới mức còn vừa ruột bóng.
    #   • bóng dò được mà phải ép XUỐNG DƯỚI font_min mới vừa → coi là bóng ẢO
    #     (floodfill/canny vớ nhầm trên nền trống — vd trang nền đen) hoặc bóng
    #     không chứa nổi câu → CỨU: bỏ bubble-fit, free-scale ở cỡ nhóm (box tự nới).
    #   • sfx/hét/giận: to nhỏ là chủ ý nghệ thuật — để tự do, chỉ ghìm trần lỏng
    #     1.8× cỡ nhóm để chống outlier vô lý.
    _BODY_KINDS = {'normal', 'narration', 'thought', 'moan', 'fear'}
    jobs_by_region = {id(_r): _k for _k, (_idx, _r, _bub, _lng) in enumerate(bubble_jobs)}
    body_regions = []
    for r in text_regions:
        r._font_kind = _classify_region(r)  # cache — dispatch() dùng lại khi chọn font
        if r._font_kind in _BODY_KINDS and r.font_size > 0:
            body_regions.append(r)
    fs_min = max(1, font_size_minimum)

    def _apply_body_target(target: int):
        """Kéo mọi vùng body về cỡ `target` (bóng thật chỉ lên tới mức còn vừa ruột;
        bóng ảo được cứu sang free-scale). Trả (nâng, ghìm, cứu) để log."""
        n_up = n_down = n_rescue = 0
        for r in body_regions:
            k = jobs_by_region.get(id(r))
            if k is None:
                # Không có job dựng lại box (nhánh nới-cao / min_rect fallback) →
                # chỉ ghìm xuống như trước; nâng lên mà không nới box thì warp lại
                # co chữ về cỡ box, không ăn.
                if r.font_size > target:
                    r.font_size = target
                    n_down += 1
                continue
            _idx, _reg, _bub, _lng = bubble_jobs[k]
            # NÂNG có trần 1.3× cỡ tự nhiên: median trang bị caption/splash to kéo
            # lên thì câu narration dài đừng phình theo quá tay (56→93 trông thô) —
            # đồng bộ là kéo GẦN nhau, không phải thổi chữ nhỏ lên gấp rưỡi.
            # Ghìm xuống thì về thẳng target (không trần).
            if target > r.font_size:
                eff_target = min(target, max(r.font_size + 1, int(r.font_size * 1.3)))
            else:
                eff_target = target
            if _bub is None:
                # Chữ tự do: box dựng lại theo fs → kéo về cỡ nhóm (nâng có trần).
                n_up += int(eff_target > r.font_size)
                n_down += int(eff_target < r.font_size)
                r.font_size = eff_target
            else:
                fit = _bubble_max_fit(_reg, _bub, _lng, eff_target)
                # Van cứu "bóng ảo" (≤50% target) CHỈ cho bóng DÒ — floodfill/canny
                # có thể vớ nhầm (vd vệt inpaint sót trên nền đen). BOX GỐC là box
                # detector THẬT: fit nhỏ hơn body trang là vì box nhỏ thật (caption
                # dải hẹp…) — cứu sang free-scale sẽ THỔI chữ lên target (52 → 118,
                # 7 dòng phủ panel) phá layout; chỉ giữ sàn đọc fs_min.
                _phantom = fit < fs_min or (
                    not getattr(_reg, '_box_fit', False)
                    and target > fs_min and fit <= max(fs_min, int(target * 0.5)))
                if _phantom:
                    # Bóng ảo / bóng-box quá bé: cứu — bỏ bubble-fit, free-scale ở
                    # cỡ nhóm (rescue về THẲNG target, không up-cap — cỡ bé kia là
                    # artifact của khung chứa, không phải cỡ tự nhiên). Bóng thật
                    # nhỏ (thì thầm) fit ~60-80% body nên không bị đụng.
                    bubble_jobs[k] = (_idx, _reg, None, _lng)
                    r.font_size = target
                    n_rescue += 1
                else:
                    n_up += int(fit > r.font_size)
                    n_down += int(fit < r.font_size)
                    r.font_size = fit
        loose_cap = int(target * 1.8)
        for r in text_regions:
            if getattr(r, '_font_kind', None) not in _BODY_KINDS and r.font_size > loose_cap:
                r.font_size = loose_cap
                n_down += 1
        return n_up, n_down, n_rescue

    def _recompute_job_boxes():
        # Tính LẠI khối chữ bằng cỡ chữ hiện hành (dst_points quyết định cỡ hiển
        # thị qua warp, nên phải dựng lại theo fs mới).
        #   • bub_in != None → khối căn giữa trong bong bóng.
        #   • bub_in == None → hộp đúng kích thước khối chữ (free scale).
        for idx, region, bub_in, lang in bubble_jobs:
            if not (0 <= idx < len(dst_points_list)):
                continue
            if bub_in is not None:
                # CHAT BOX → căn giữa trong ruột bóng.
                dst_points_list[idx] = _centered_text_block(
                    region.font_size, region.translation, bub_in, lang,
                    img.shape[1], img.shape[0])
            else:
                # Chữ tự do (ngang/dọc) → hộp khít khối chữ, neo tâm vùng gốc.
                dp = _scaled_box_for_floor(region, region.font_size, img)
                if dp is not None:
                    dst_points_list[idx] = dp

    # ── Tách các khối chữ ĐÈ NHAU ─────────────────────────────────────────────
    # Hai khối chồng nhau >12% (vùng nhỏ hơn) → ĐẨY RA XA theo trục đè ít hơn
    # (mỗi khối dịch nửa độ chồng), đảm bảo tách kể cả khi gần cùng tâm (thu nhỏ
    # không tách được). Nếu bị kẹt ở mép ảnh thì thu nhỏ nhẹ. Lặp đến khi hết đè.
    iw, ih = img.shape[1], img.shape[0]

    def _aabb(dp):
        p = np.asarray(dp, dtype=np.float64).reshape(-1, 2)
        return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()

    def _shift(idx, dx, dy):
        # Dịch NGUYÊN khối (rigid): clamp delta để khối luôn nằm trong ảnh, KHÔNG
        # clip từng điểm (clip rời điểm sẽ làm khối bẹp về 0 chiều → r_orig NaN → vỡ).
        shp = np.asarray(dst_points_list[idx]).shape
        dp = np.asarray(dst_points_list[idx], dtype=np.float64).reshape(-1, 2)
        x1, y1, x2, y2 = dp[:, 0].min(), dp[:, 1].min(), dp[:, 0].max(), dp[:, 1].max()
        dx = min(dx, (iw - 1) - x2) if dx > 0 else max(dx, -x1)
        dy = min(dy, (ih - 1) - y2) if dy > 0 else max(dy, -y1)
        dp[:, 0] += dx
        dp[:, 1] += dy
        dst_points_list[idx] = dp.astype(np.int64).reshape(shp)

    def _shrink(idx, f):
        shp = np.asarray(dst_points_list[idx]).shape
        dp = np.asarray(dst_points_list[idx], dtype=np.float64).reshape(-1, 2)
        c = dp.mean(axis=0)
        dp = (dp - c) * f + c
        dst_points_list[idx] = dp.astype(np.int64).reshape(shp)

    n_box = len(dst_points_list)

    def _overlap_pairs():
        """Các cặp (i, j) CHỒNG nhau > 6% diện tích vùng nhỏ hơn (đọc aabb tươi)."""
        out = []
        for i in range(n_box):
            for j in range(i + 1, n_box):
                ax1, ay1, ax2, ay2 = _aabb(dst_points_list[i])
                bx1, by1, bx2, by2 = _aabb(dst_points_list[j])
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)
                if ox <= 0 or oy <= 0:
                    continue
                amin = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                if amin <= 0 or ox * oy <= 0.06 * amin:
                    continue
                out.append((i, j))
        return out

    def _push_apart(i, j):
        # ĐẶT theo THỨ TỰ ĐỌC quanh tâm chung: i < j = đọc trước → i sang TRÁI/TRÊN,
        # j sang PHẢI/DƯỚI. Đặt thẳng (không chỉ nhích) để SỬA cả khi gốc bị đảo (vd
        # CJK dọc đọc phải→trái: 感觉 ở phải, 如何 ở trái → hoán "Cảm giác" trái,
        # "thế nào" phải). Tách theo TRỤC ĐÈ ÍT HƠN (mỗi khối dịch nửa độ chồng).
        # NGOẠI LỆ cột free-text (_col_keep): trang CJK dọc đọc PHẢI→TRÁI có chủ
        # đích — giữ TƯƠNG QUAN HÌNH HỌC hiện có (khối đang bên nào tách về bên
        # đó), KHÔNG ép theo thứ tự đọc kẻo lật layout phải→trái thành trái→phải.
        ax1, ay1, ax2, ay2 = _aabb(dst_points_list[i])
        bx1, by1, bx2, by2 = _aabb(dst_points_list[j])
        ox = min(ax2, bx2) - max(ax1, bx1)
        oy = min(ay2, by2) - max(ay1, by1)
        ci_x, cj_x = (ax1 + ax2) / 2, (bx1 + bx2) / 2
        ci_y, cj_y = (ay1 + ay2) / 2, (by1 + by2) / 2
        wi, wj = ax2 - ax1, bx2 - bx1
        hi, hj = ay2 - ay1, by2 - by1
        gap = 6
        try:
            keep_geom = (getattr(text_regions[i], '_col_keep', False)
                         and getattr(text_regions[j], '_col_keep', False))
        except Exception:
            keep_geom = False
        if keep_geom:
            # Hoán vai để "i" luôn là khối đang ở TRÁI/TRÊN theo hình học.
            if (ox <= oy and ci_x > cj_x) or (ox > oy and ci_y > cj_y):
                i, j = j, i
                ci_x, cj_x, ci_y, cj_y = cj_x, ci_x, cj_y, ci_y
                wi, wj, hi, hj = wj, wi, hj, hi
        if ox <= oy:
            cc = (ci_x + cj_x) / 2
            _shift(i, (cc - gap / 2 - wi / 2) - ci_x, 0)
            _shift(j, (cc + gap / 2 + wj / 2) - cj_x, 0)
        else:
            cc = (ci_y + cj_y) / 2
            _shift(i, 0, (cc - gap / 2 - hi / 2) - ci_y)
            _shift(j, 0, (cc + gap / 2 + hj / 2) - cj_y)

    # SÀN THU NHỎ: không thu khối quá nhỏ kẻo chữ không đọc được (env
    # MIT_OVERLAP_SHRINK_FLOOR, mặc định 0.5 = thu tối đa còn 50%; đặt 1.0 = TẮT thu,
    # chỉ đẩy như cũ). Kẹp trong [0.3, 1.0].
    try:
        floor = float(os.environ.get("MIT_OVERLAP_SHRINK_FLOOR", "") or 0.5)
    except (TypeError, ValueError):
        floor = 0.5
    floor = min(1.0, max(0.3, floor))

    def _separate_blocks():
        """Tách khối đè: 1) ĐẨY ra xa (giữ cỡ chữ); 2) còn kẹt → THU NHỎ hình học
        dần tới sàn rồi đẩy tiếp. Trả (số lần đẩy, số lần thu) để vòng phản hồi
        bên dưới biết trang có CHẬT không."""
        moves = shrinks = 0
        for _ in range(8):
            pairs = _overlap_pairs()
            if not pairs:
                break
            for (i, j) in pairs:
                _push_apart(i, j)
                moves += 1
        _scale = [1.0] * n_box
        for _ in range(12):
            pairs = _overlap_pairs()
            if not pairs:
                break
            shrank = False
            for (i, j) in pairs:
                for k in (i, j):
                    if _scale[k] > floor:
                        _scale[k] *= 0.85
                        _shrink(k, 0.85)
                        shrinks += 1
                        shrank = True
            # đẩy lại để dồn phần dư sau khi đã nhỏ đi
            for (i, j) in _overlap_pairs():
                _push_apart(i, j)
                moves += 1
            if not shrank:
                break  # mọi khối còn đè đã chạm sàn → không thu thêm được
        return moves, shrinks

    # ── Điều phối: đồng bộ cỡ body → dựng box → tách đè, có VÒNG PHẢN HỒI ──────
    # Tách đè phải THU NHỎ nghĩa là trang CHẬT — nhưng thu hình học từng khối lẻ
    # (khối 85%, khối 72%…) lại phá đồng bộ cỡ chữ vừa làm. Chuẩn typesetting:
    # hạ ĐỒNG LOẠT cỡ body một nấc rồi xếp lại từ đầu — cả trang vẫn MỘT cỡ.
    if len(body_regions) >= 2:
        # Vùng box-fit (box gốc) bị box dò BÓP cỡ chữ — cỡ đã fit KHÔNG phản ánh cỡ
        # tự nhiên. Median lấy _natural_fs đã lưu cho các vùng đó (bóng thật/tự do
        # giữ cỡ hiện có), kẻo vài cột hẹp kéo sập body target của cả trang.
        sizes = [int(getattr(r, '_natural_fs', 0)) or r.font_size for r in body_regions]
        # Median trên các cỡ "lành" (≥ font_min) — cỡ tí hon từ bóng ảo/box lỗi
        # không được phép kéo sập chuẩn của cả trang.
        healthy = [s for s in sizes if s >= fs_min] or sizes
        target = max(fs_min, int(round(float(np.median(healthy)))))
        for _round in range(2):
            n_up, n_down, n_rescue = _apply_body_target(target)
            logger.info(f'[font-normalize] body target={target} ({len(body_regions)} vùng) '
                        f'→ nâng {n_up}, ghìm {n_down}, cứu {n_rescue} bóng ảo.')
            _recompute_job_boxes()
            moves, shrinks = _separate_blocks()
            logger.info(f'[separate] {n_box} khối, đẩy {moves} lần, thu nhỏ {shrinks} lần để tách đè.')
            if shrinks == 0 or target <= fs_min or _round == 1:
                break
            target = max(fs_min, int(target * 0.85))
            logger.info(f'[font-normalize] trang chật → hạ body target={target}, xếp lại toàn trang.')
    else:
        _recompute_job_boxes()
        moves, shrinks = _separate_blocks()
        logger.info(f'[separate] {n_box} khối, đẩy {moves} lần, thu nhỏ {shrinks} lần để tách đè.')

    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False
    ) -> np.ndarray:

    text_render.set_font(font_path)
    # Gộp các vùng [cont] (một câu nguồn bị detector cắt thành nhiều vùng) TRƯỚC
    # khi lọc — vùng cont phải còn mặt ở đây để union hình học vào vùng anchor.
    _merge_cont_regions(text_regions)
    # Bỏ vùng không có gì để render: rỗng hoặc chỉ gồm ký tự vô hình (ZWJ translator
    # trả cho watermark/[cont]). Không lọc → chúng được fit cỡ chữ to (1 ký tự /
    # box rộng) kéo lệch chuẩn cỡ chữ toàn trang + tốn lượt render thừa.
    text_regions = [r for r in text_regions if _meaningful_translation(r.translation)]

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    # Đa dạng font theo loại thoại — chọn font riêng cho mỗi vùng (lưu ý 2).
    font_map = _resolve_font_paths(font_path)

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        # Đổi font theo loại thoại trước khi render vùng này (get_cached_font cache nên rẻ).
        # Ưu tiên kind đã phân loại ở bước đồng bộ cỡ chữ (nhất quán size ↔ font).
        kind = getattr(region, '_font_kind', None) or _classify_region(region)
        chosen_font = font_map.get(kind, font_path)
        text_render.set_font(chosen_font)
        logger.info(f'[font] "{(region.get_translation_for_rendering() or "")[:16]}" '
                    f'→ {kind} ({os.path.basename(chosen_font) if chosen_font else "default"})')
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    # ── Sample ACTUAL image pixels at render location ──────────────────────────
    # fg_bg_compare chỉ dựa vào màu OCR detect (bị ảnh hưởng bởi halftone manga).
    # Sample pixel thực tế sau inpaint để biết chính xác nền tối hay sáng.
    if bg is not None:
        try:
            pts = dst_points[0]  # shape (4, 2)
            minx = int(max(0, pts[:, 0].min()))
            maxx = int(min(img.shape[1], pts[:, 0].max()))
            miny = int(max(0, pts[:, 1].min()))
            maxy = int(min(img.shape[0], pts[:, 1].max()))
            if maxx > minx and maxy > miny:
                actual_brightness = float(np.mean(img[miny:maxy, minx:maxx]))
                fg_brightness = float(np.mean(fg))
                triggered = actual_brightness < 210 and fg_brightness < 200
                logger.info(
                    f'[stroke-debug] text="{region.get_translation_for_rendering()[:30]}" '
                    f'actual_bg={actual_brightness:.1f} fg_bright={fg_brightness:.1f} '
                    f'fg_in={fg} bg_in={bg} '
                    f'-> {"WHITE_STROKE" if triggered else "no_change"}'
                )
                if triggered:
                    fg = (0, 0, 0)           # chữ đen
                    bg = (255, 255, 255)     # stroke trắng dày (như bản gốc tiếng Trung)
        except Exception as e:
            logger.info(f'[stroke-debug] exception: {e}')

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    # Guard: dst_points suy biến (cao/rộng = 0) → chia 0 → r_orig NaN/inf → vỡ ở
    # các phép int(... r_orig ...). Ép về tỉ lệ an toàn 1.0 và bỏ qua nếu quá nhỏ.
    with np.errstate(divide='ignore', invalid='ignore'):
        r_orig = float(np.mean(norm_h / norm_v))
    if not np.isfinite(r_orig) or r_orig <= 0:
        r_orig = 1.0
    if float(np.mean(norm_h)) < 1.0 or float(np.mean(norm_v)) < 1.0:
        return img  # khối quá nhỏ/suy biến — bỏ qua, không render (tránh crash)

    # If configuration is set to non-automatic mode, use configuration to determine direction directly
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    #print(f"Region text: {region.text}, forced_direction: {forced_direction}, render_horizontally: {render_horizontally}")

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    if temp_box is None:
        return img
    h, w, _ = temp_box.shape
    if h == 0 or w == 0:
        return img  # canvas chữ suy biến (vd toàn-0 → add_color cắt còn 0) → bỏ qua, tránh chia 0
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    # Cap h_ext/w_ext to prevent enormous numpy array allocation when r_orig is
    # near-zero (very narrow region) or very large (very wide region).
    _MAX_BOX_SIDE = 4096

    if region.horizontal:  
        if r_temp > r_orig:   
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0
            h_ext = min(h_ext, max((_MAX_BOX_SIDE - h) // 2, 0))  # guard
            if h_ext >= 0:  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                box = temp_box.copy()  
        else:   
            w_ext = int((h * r_orig - w) // 2)
            w_ext = min(w_ext, max((_MAX_BOX_SIDE - w) // 2, 0))  # guard
            if w_ext >= 0:  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                box[0:h, 0:w] = temp_box  
            else:  
                box = temp_box.copy()  
    else:  
        if r_temp > r_orig:   
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0
            h_ext = min(h_ext, max((_MAX_BOX_SIDE - h) // 2, 0))  # guard
            if h_ext >= 0:   
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                box[0:h, 0:w] = temp_box  
            else:   
                box = temp_box.copy()   
        else:   
            w_ext = int((h * r_orig - w) / 2)
            w_ext = min(w_ext, max((_MAX_BOX_SIDE - w) // 2, 0))  # guard
            if w_ext >= 0:  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                box = temp_box.copy()   
    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)

async def dispatch_eng_render_pillow(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/NotoSansMonoCJK-VF.ttf.ttc')
    text_render.set_font(font_path)

    return render_textblock_list_eng_pillow(font_path, img_canvas, text_regions, original_img=original_img, downscale_constraint=0.95)
