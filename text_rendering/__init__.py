from typing import List, Union
from utils import Quadrilateral
import numpy as np
import cv2

from utils import findNextPowerOf2, color_difference
from detection.ctd_utils import TextBlock
from . import text_render
from .text_render_eng import render_textblock_list_eng


LANGAUGE_ORIENTATION_PRESETS = {
    'CHS': 'auto',
    'CHT': 'auto',
    'CSY': 'h',
    'NLD': 'h',
    'ENG': 'h',
    'FRA': 'h',
    'DEU': 'h',
    'HUN': 'h',
    'ITA': 'h',
    'JPN': 'auto',
    'KOR': 'auto',
    'PLK': 'h',
    'PTB': 'h',
    'ROM': 'h',
    'RUS': 'h',
    'ESP': 'h',
    'TRK': 'h',
    'VIN': 'h',
}

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 15:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

async def dispatch(
    img_canvas: np.ndarray,
    text_mag_ratio: np.integer,
    translated_sentences: List[str],
    text_regions: List[TextBlock],
    text_direction_overwrite: str,
    target_language: str,
    font_size_offset: int = 0,
    render_mask: np.ndarray = None
) -> np.ndarray:
    for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)):
        print(f'text: {region.get_text()} \n trans: {trans_text}')
        if not trans_text:
            continue

        majority_dir = None
        angle_changed = False
        if text_direction_overwrite in ['h', 'v']:
            majority_dir = text_direction_overwrite
        elif target_language in LANGAUGE_ORIENTATION_PRESETS:
            majority_dir = LANGAUGE_ORIENTATION_PRESETS[target_language]
        if majority_dir not in ['h', 'v']:
            if region.vertical:
                majority_dir = 'v'
                region.angle += 90
                angle_changed = True
            else:
                majority_dir = 'h'

        fg, bg = region.get_font_colors()
        fg, bg = fg_bg_compare(fg, bg)
        font_size = region.font_size
        font_size = round(font_size)
        if not isinstance(font_size, int):
            font_size = int(font_size)

        # for i, text in enumerate(region.text):
        #     textline = Quadrilateral(np.array(region.lines[i]), text, 1, region.fg_r, region.fg_g, region.fg_b, region.bg_r, region.bg_g, region.bg_b)
        #     img_canvas = render(img_canvas, font_size, text_mag_ratio, trans_text, textline, majority_dir, fg, bg, False, font_size_offset)

        img_canvas = render(img_canvas, font_size, text_mag_ratio, trans_text, region, majority_dir, fg, bg, True, font_size_offset, render_mask)
        if angle_changed:
            region.angle -= 90
    return img_canvas

def render(
    img_canvas,
    font_size,
    text_mag_ratio,
    trans_text,
    region,
    majority_dir,
    fg,
    bg,
    is_ctd,
    font_size_offset: int = 0,
    render_mask: np.ndarray = None
):
    # round font_size to fixed powers of 2, so later LRU cache can work
    font_size_enlarged = findNextPowerOf2(font_size) * text_mag_ratio
    enlarge_ratio = font_size_enlarged / font_size
    font_size = font_size_enlarged
    #enlarge_ratio = 1
    while True:
        if is_ctd:
            enlarged_w = round(enlarge_ratio * (region.xyxy[2] - region.xyxy[0]))
            enlarged_h = round(enlarge_ratio * (region.xyxy[3] - region.xyxy[1]))
        else:
            enlarged_w = round(enlarge_ratio * region.aabb.w)
            enlarged_h = round(enlarge_ratio * region.aabb.h)
        rows = enlarged_h // (font_size * 1.3)
        cols = enlarged_w // (font_size * 1.3)
        if rows * cols < len(trans_text):
            enlarge_ratio *= 1.1
            continue
        break
    font_size += font_size_offset
    print('font_size:', font_size)
    if majority_dir == 'h':
        temp_box = text_render.put_text_horizontal(
            font_size,
            enlarge_ratio * 1.0,
            trans_text,
            enlarged_w,
            fg,
            bg
        )
    else:
        temp_box = text_render.put_text_vertical(
            font_size,
            enlarge_ratio * 1.0,
            trans_text,
            enlarged_h,
            fg,
            bg
        )
    h, w, _ = temp_box.shape
    r_prime = w / h

    r = region.aspect_ratio
    if is_ctd and majority_dir != 'v':
        r = 1 / r

    w_ext = 0
    h_ext = 0
    if r_prime > r:
        h_ext = int(w / (2 * r) - h / 2)
        box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
        box[h_ext:h + h_ext, 0:w] = temp_box
    else:
        w_ext = int((h * r - w) / 2)
        box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        box[0:h, w_ext:w_ext+w] = temp_box
    #h_ext += region_ext
    #w_ext += region_ext

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)
    if is_ctd:
        dst_points = region.min_rect()
        if majority_dir == 'v':
            dst_points = dst_points[:, [3, 0, 1, 2]]
    else:
        dst_points = region.pts

    if render_mask is not None:
        # set render_mask to 1 for the region that is inside dst_points
        cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = np.clip(cv2.warpPerspective(box, M, (img_canvas.shape[1], img_canvas.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
    canvas_region = rgba_region[:, :, 0: 3]
    mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
    img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img_canvas

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: Union[List[TextBlock], List[Quadrilateral]], translated_sentences: List[str], font_path: str = 'fonts/comic shanns 2.ttf') -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if isinstance(text_regions[0], Quadrilateral):
        blk_list = []
        for region, tr in zip(text_regions, translated_sentences):
            x1 = np.min(region.pts[:, 0])
            x2 = np.max(region.pts[:, 0])
            y1 = np.min(region.pts[:, 1])
            y2 = np.max(region.pts[:, 1])
            font_size = region.font_size * 0.75        # default detector generate larger text polygons in my exp
            angle = np.rad2deg(region.angle) - 90
            if abs(angle) < 3:
                angle = 0
            blk = TextBlock([x1, y1, x2, y2], lines=[region.pts], translation=tr, angle=angle, font_size=font_size)
            blk_list.append(blk)
        return render_textblock_list_eng(img_canvas, blk_list, font_path, size_tol=1.2, original_img=original_img, downscale_constraint=0.5)

    for blk, tr in zip(text_regions, translated_sentences):
        blk.translation = tr

    return render_textblock_list_eng(img_canvas, text_regions, font_path, size_tol=1.2, original_img=original_img, downscale_constraint=0.8)
