import os
import cv2
import numpy as np
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

            # ── Step 2: if still can't fit → expand HEIGHT only (never width) ──
            if needed_rows > rows_capacity(fit_size):
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
                        # Narrow region (manga bubble): grow downward, anchor at top
                        poly = affinity.scale(poly, xfact=1.0, yfact=scale_y, origin=(minx, miny))
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

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
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
    r_orig = np.mean(norm_h / norm_v)

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
