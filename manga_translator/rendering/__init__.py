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
    "sfx":       "MTO COMIC 2.ttf",          # SFX / rên — ngoài khung (có sẵn)
    "shout":     "MTO Damn Noisy Kids.ttf",  # thoại hét               (Việt hoá, đã tải)
    "narration": "MTO Chaney.ttf",           # thoại dẫn truyện        (Việt hoá, đã tải; cần classifier mới dùng)
}
# Hét: dấu '!' lặp (!!), hoặc '?!' / '!?' (kể cả fullwidth). KHÔNG khớp 1 dấu '!'
# đơn lẻ vì thoại thường tiếng Việt cũng hay kết thúc bằng '!'.
_SHOUT_RE = re.compile(r"[!！]{2,}|[?？][!！]|[!！][?？]")


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

    # Hét: dấu trong chữ HOẶC LLM nói shout HOẶC burst gai-nền-đen (nhấn mạnh).
    letters = [c for c in text if c.isalpha()]
    all_caps_short = bool(letters) and len(text) <= 14 and all(c.isupper() for c in letters)
    if _SHOUT_RE.search(text) or all_caps_short or llm == "shout" or kind == "burst":
        return "shout"
    # Rên / tượng thanh: tin LLM (đoán theo nội dung) → font Comic 2.
    if llm in ("moan", "sfx"):
        return "sfx"
    # Dẫn truyện: khung chữ nhật HOẶC LLM nói narration.
    if kind == "rect" or llm == "narration":
        return "narration"
    # Suy nghĩ: CHỈ theo hình dạng mây (LLM hay nhầm nói→nghĩ nên không tin LLM 'thought').
    if kind == "cloud":
        return "thought"
    # Chữ ngoài bong bóng → coi như SFX/rên.
    if not _region_is_bubble(region):
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
    th = int(n * fs + (n - 1) * fs * 0.14)
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

    dst_points_list = []
    bubble_jobs = []  # (index, region, bub_in, lang) để tính lại khối sau khi cap cỡ chữ
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

        # ── Detect if inside a styled speech bubble ──
        # High contrast (white-on-black / black-on-white) OR light background
        try:
            fg_col, bg_col = region.get_font_colors()
            bg_avg = np.mean(bg_col)
            contrast = color_difference(fg_col, bg_col)
            is_bubble = bg_avg > 140 or contrast > 100
        except Exception:
            is_bubble = True  # safer default

        dst_points = None
        single_axis_expanded = False

        if region.horizontal:
            used_rows = max(1, len(region.texts))

            # ── Lưu ý #1: căn theo BONG BÓNG thật ──────────────────────────────
            # Nếu dò được ruột bong bóng, dùng nó làm vùng fit + đặt chữ (căn giữa,
            # chữ to tự nhiên) thay vì vùng text unclip (ôm sát → chữ bị co nhỏ).
            bubble_rect = _bubble_interior_rect(img, region) if is_bubble else None
            bub_in = None  # (in_x1, in_y1, in_x2, in_y2, cx, cy) vùng đặt chữ (đã giới hạn)
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

            # ── Lưu ý #1 (chuẩn): đặt KHỐI CHỮ ở GIỮA bong bóng, cỡ TỰ NHIÊN ──
            # render() warp chữ để LẤP ĐẦY dst_points → nếu dst_points = cả ruột
            # bong bóng thì chữ phình quá to & đè bóng cạnh. Vì vậy đặt dst_points =
            # đúng khối chữ ở target_font_size, căn giữa bong bóng (không lấp đầy).
            if bub_in is not None:
                try:
                    lang = getattr(region, "target_lang", "en_US")
                    dst_points = _centered_text_block(
                        target_font_size, region.translation, bub_in, lang,
                        img.shape[1], img.shape[0])
                    # Ghi nhận để TÍNH LẠI khối sau khi cap cỡ chữ (cap mới hiệu lực).
                    bubble_jobs.append((len(dst_points_list), region, bub_in, lang))
                    logger.info(f'[bubble-fit] "{region.get_translation_for_rendering()[:18]}" '
                                f'fs={target_font_size} giữa bóng')
                except Exception:
                    pass

        if region.vertical:
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

        dst_points_list.append(dst_points)
        region.font_size = int(target_font_size)

    # ── Chuẩn hoá cỡ chữ toàn trang (lưu ý 3: chữ to chữ nhỏ chênh lệch) ──
    # Ghìm các vùng có cỡ chữ to bất thường về gần trung vị để cả trang đỡ chênh.
    # CHỈ thu NHỎ (không phóng to) → an toàn, không gây tràn bong bóng. Cho phép
    # vẫn lớn hơn trung vị 25% để thoại hét/nhấn mạnh vẫn nổi bật.
    sizes = [r.font_size for r in text_regions if r.font_size > 0]
    if len(sizes) >= 3:
        med = float(np.median(sizes))
        cap = max(int(round(med * 0.95)), max(1, font_size_minimum))
        n_capped = 0
        for r in text_regions:
            if r.font_size > cap:
                r.font_size = cap
                n_capped += 1
        if n_capped:
            logger.info(f'[font-normalize] median={med:.0f} cap={cap} → ghìm {n_capped} vùng chữ to.')

    # Tính LẠI khối chữ bubble-fit bằng cỡ chữ ĐÃ CAP (để cap có hiệu lực thật:
    # dst_points quyết định cỡ hiển thị qua warp, nên phải dựng lại theo fs mới).
    for idx, region, bub_in, lang in bubble_jobs:
        if 0 <= idx < len(dst_points_list):
            dst_points_list[idx] = _centered_text_block(
                region.font_size, region.translation, bub_in, lang,
                img.shape[1], img.shape[0])

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
    _sep_moves = 0
    for _ in range(8):
        moved = False
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
                # ĐẶT theo THỨ TỰ ĐỌC quanh tâm chung: i < j = đọc trước → i sang
                # TRÁI/TRÊN, j sang PHẢI/DƯỚI. Đặt thẳng (không chỉ nhích) để SỬA cả
                # khi gốc bị đảo (vd CJK dọc đọc phải→trái: 感觉 ở phải, 如何 ở trái →
                # phải hoán lại thành "Cảm giác" trái, "thế nào" phải).
                ci_x, cj_x = (ax1 + ax2) / 2, (bx1 + bx2) / 2
                ci_y, cj_y = (ay1 + ay2) / 2, (by1 + by2) / 2
                wi, wj = ax2 - ax1, bx2 - bx1
                hi, hj = ay2 - ay1, by2 - by1
                gap = 6
                if ox <= oy:
                    cc = (ci_x + cj_x) / 2
                    _shift(i, (cc - gap / 2 - wi / 2) - ci_x, 0)
                    _shift(j, (cc + gap / 2 + wj / 2) - cj_x, 0)
                else:
                    cc = (ci_y + cj_y) / 2
                    _shift(i, 0, (cc - gap / 2 - hi / 2) - ci_y)
                    _shift(j, 0, (cc + gap / 2 + hj / 2) - cj_y)
                moved = True
                _sep_moves += 1
        if not moved:
            break
    else:
        # Sau 8 vòng vẫn còn đè (kẹt mép ảnh) → thu nhỏ các cặp còn chồng.
        for i in range(n_box):
            for j in range(i + 1, n_box):
                ax1, ay1, ax2, ay2 = _aabb(dst_points_list[i])
                bx1, by1, bx2, by2 = _aabb(dst_points_list[j])
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)
                if ox > 0 and oy > 0:
                    _shrink(i, 0.8); _shrink(j, 0.8)

    logger.info(f'[separate] {n_box} khối, đã đẩy {_sep_moves} lần để tách đè.')
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
    text_regions = list(filter(lambda region: region.translation, text_regions))

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
        kind = _classify_region(region)
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
