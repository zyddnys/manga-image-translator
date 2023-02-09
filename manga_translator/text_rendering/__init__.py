from typing import List
import numpy as np
import cv2
import os

from . import text_render
from .text_render_eng import render_textblock_list_eng
from .ballon_extractor import extract_ballon_region
from ..utils import findNextPowerOf2, color_difference
from ..detection.ctd_utils import TextBlock


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

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 15:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    text_mag_ratio: np.integer,
    text_direction: str = 'auto',
    font_path: str = '',
    font_size_offset: int = 0,
    original_img: np.ndarray = None,
    render_mask: np.ndarray = None,
    ) -> np.ndarray:

    text_render.set_font(font_path)

    for region in text_regions:
        print(f'text: {region.get_text()} \n trans: {region.translation}')
        if not region.translation:
            continue

        majority_dir = None
        angle_changed = False
        if text_direction in ['h', 'v']:
            majority_dir = text_direction
        elif region.target_lang in LANGAUGE_ORIENTATION_PRESETS:
            majority_dir = LANGAUGE_ORIENTATION_PRESETS[region.target_lang]
        if majority_dir not in ['h', 'v']:
            if region.vertical:
                majority_dir = 'v'
                # TODO: Make this unnecessary
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

        img = render(img, region, font_size, text_mag_ratio, majority_dir, fg, bg, original_img, font_size_offset, render_mask)
        if angle_changed:
            region.angle -= 90
    return img

def render(
    img,
    region: TextBlock,
    font_size,
    text_mag_ratio,
    majority_dir,
    fg, bg,
    original_img,
    font_size_offset: int = 0,
    render_mask: np.ndarray = None,
):
    # round font_size to fixed powers of 2, so later LRU cache can work
    font_size_enlarged = findNextPowerOf2(font_size) * text_mag_ratio
    enlarge_ratio = font_size_enlarged / font_size
    font_size = font_size_enlarged
    while True:
        enlarged_w = round(enlarge_ratio * (region.xyxy[2] - region.xyxy[0]))
        enlarged_h = round(enlarge_ratio * (region.xyxy[3] - region.xyxy[1]))
        rows = enlarged_h // (font_size * 1.3)
        cols = enlarged_w // (font_size * 1.3)
        if rows * cols < len(region.translation):
            enlarge_ratio *= 1.1
            continue
        break
    font_size += font_size_offset
    print('font_size:', font_size)

    # TODO: Add ballon extractor
    # bounding_rect = region.bounding_rect()
    # # non-dl textballon segmentation
    # enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, 3)
    # ballon_mask, ballon_area, xyxy = extract_ballon_region(original_img, bounding_rect, enlarge_ratio=enlarge_ratio)

    if majority_dir == 'h':
        temp_box = text_render.put_text_horizontal(
            font_size,
            enlarge_ratio * 1.0,
            region.translation,
            enlarged_w,
            fg,
            bg,
        )
    else:
        temp_box = text_render.put_text_vertical(
            font_size,
            enlarge_ratio * 1.0,
            region.translation,
            enlarged_h,
            fg,
            bg,
        )
    h, w, _ = temp_box.shape
    r_prime = w / h

    r = region.aspect_ratio
    if majority_dir != 'v':
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
    dst_points = region.min_rect()
    if majority_dir == 'v':
        dst_points = dst_points[:, [3, 0, 1, 2]]

    if render_mask is not None:
        # set render_mask to 1 for the region that is inside dst_points
        cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = np.clip(cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
    canvas_region = rgba_region[:, :, 0: 3]
    mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
    img = np.clip((img.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '') -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = 'fonts/comic shanns 2.ttf'
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, size_tol=1.2, original_img=original_img, downscale_constraint=0.8)
