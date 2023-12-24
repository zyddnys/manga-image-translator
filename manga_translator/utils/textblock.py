import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPoint
from functools import cached_property
import copy
import re
import py3langid as langid

from .generic import color_difference, is_right_to_left_char, is_valuable_char
# from ..detection.ctd_utils.utils.imgproc_utils import union_area, xywh2xyxypoly

# LANG_LIST = ['eng', 'ja', 'unknown']
# LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

# determines render direction
LANGUAGE_ORIENTATION_PRESETS = {
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
    'UKR': 'h',
    'VIN': 'h',
    'ARA': 'hr', # horizontal reversed (right to left)
}

class TextBlock(object):
    """
    Object that stores a block of text made up of textlines.
    """
    def __init__(self, lines: List,
                 texts: List[str] = None,
                 language: str = 'unknown',
                 font_size: float = -1,
                 angle: int = 0,
                 translation: str = "",
                 fg_color: Tuple[float] = (0, 0, 0),
                 bg_color: Tuple[float] = (0, 0, 0),
                 line_spacing = 1.,
                 letter_spacing = 1.,
                 font_family: str = "",
                 bold: bool = False,
                 underline: bool = False,
                 italic: bool = False,
                 direction: str = 'auto',
                 alignment: str = 'auto',
                 rich_text: str = "",
                 _bounding_rect: List = None,
                 default_stroke_width = 0.2,
                 font_weight = 50,
                 source_lang: str = "",
                 target_lang: str = "",
                 opacity: float = 1.,
                 shadow_radius: float = 0.,
                 shadow_strength: float = 1.,
                 shadow_color: Tuple = (0, 0, 0),
                 shadow_offset: List = [0, 0],
                 prob: float = 1,
                 **kwargs) -> None:
        self.lines = np.array(lines, dtype=np.int32)
        # self.lines.sort()
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.texts = texts if texts is not None else []
        self.text = texts[0]
        for txt in texts[1:] :
            first_cjk = '\u3000' <= self.text[-1] <= '\u9fff'
            second_cjk = '\u3000' <= txt[0] <= '\u9fff'
            if first_cjk or second_cjk :
                self.text += txt
            else :
                self.text += ' ' + txt
        self.prob = prob

        self.translation = translation

        self.fg_colors = fg_color
        self.bg_colors = bg_color

        # self.stroke_width = stroke_width
        self.font_family: str = font_family
        self.bold: bool = bold
        self.underline: bool = underline
        self.italic: bool = italic
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self._alignment = alignment
        self._source_lang = source_lang
        self.target_lang = target_lang

        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.adjust_bg_color = True

        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset

    @cached_property
    def xyxy(self):
        """Coordinates of the bounding box"""
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return np.array([x1, y1, x2, y2]).astype(np.int32)

    @cached_property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([x1, y1, x2-x1, y2-y1]).astype(np.int32)

    @cached_property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    @cached_property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @cached_property
    def unrotated_min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.clip(0).reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def polygon_aspect_ratio(self) -> float:
        """width / height"""
        polygons = self.unrotated_polygons.reshape(-1, 4, 2)
        middle_pts = (polygons[:, [1, 2, 3, 0]] + polygons) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        return np.mean(norm_h / norm_v)

    @cached_property
    def unrotated_size(self) -> Tuple[int, int]:
        """Returns width and height of unrotated bbox"""
        middle_pts = (self.min_rect[:, [1, 2, 3, 0]] + self.min_rect) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3])
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0])
        return norm_h, norm_v

    @cached_property
    def aspect_ratio(self) -> float:
        """width / height"""
        return self.unrotated_size[0] / self.unrotated_size[1]

    @property
    def polygon_object(self) -> Polygon:
        min_rect = self.min_rect[0]
        return MultiPoint([tuple(min_rect[0]), tuple(min_rect[1]), tuple(min_rect[2]), tuple(min_rect[3])]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    @property
    def real_area(self) -> float:
        lines = self.lines.reshape((-1, 2))
        return MultiPoint([tuple(l) for l in lines]).convex_hull.area
    
    def normalized_width_list(self) -> List[float]:
        polygons = self.unrotated_polygons
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).sum())
        width_list = np.array(width_list)
        width_list = width_list / np.sum(width_list)
        return width_list.tolist()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self):
        blk_dict = copy.deepcopy(vars(self))
        return blk_dict

    def get_transformed_region(self, img: np.ndarray, line_idx: int, textheight: int, maxwidth: int = None) -> np.ndarray:
        src_pts = np.array(self.lines[line_idx], dtype=np.float64)

        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        ratio = np.linalg.norm(vec_v) / np.linalg.norm(vec_h)

        if ratio < 1:
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
        else:
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if maxwidth is not None:
            h, w = region.shape[: 2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))
        return region

    @property
    def source_lang(self):
        if not self._source_lang:
            self._source_lang = langid.classify(self.text)[0]
        return self._source_lang

    def get_translation_for_rendering(self):
        text = self.translation
        if self.direction.endswith('r'):
            # The render direction is right to left so left-to-right
            # text/number chunks need to be reversed to look normal.

            text_list = list(text)
            l2r_idx = -1

            def reverse_sublist(l, i1, i2):
                delta = i2 - i1
                for j1 in range(i1, i2 - delta // 2):
                    j2 = i2 - (j1 - i1) - 1
                    l[j1], l[j2] = l[j2], l[j1]

            for i, c in enumerate(text):
                if not is_right_to_left_char(c) and is_valuable_char(c):
                    if l2r_idx < 0:
                        l2r_idx = i
                elif l2r_idx >= 0 and i - l2r_idx > 1:
                    # Reverse left-to-right characters for correct rendering
                    reverse_sublist(text_list, l2r_idx, i)
                    l2r_idx = -1
            if l2r_idx >= 0 and i - l2r_idx > 1:
                reverse_sublist(text_list, l2r_idx, len(text_list))

            text = ''.join(text_list)
        return text

    @property
    def is_bulleted_list(self):
        """
        A determining factor of whether we should be sticking to the strict per textline
        text distribution when rendering.
        """
        if len(self.texts) <= 1:
            return False

        bullet_regexes = [
            r'[^\w\s]', # ○ ... ○ ...
            r'[\d]+\.', # 1. ... 2. ...
            r'[QA]:', # Q: ... A: ...
        ]
        bullet_type_idx = -1
        for line_text in self.texts:
            for i, breg in enumerate(bullet_regexes):
                if re.search(r'(?:[\n]|^)((?:' + breg + r')[\s]*)', line_text):
                    if bullet_type_idx >= 0 and bullet_type_idx != i:
                        return False
                    bullet_type_idx = i
        return bullet_type_idx >= 0

    def set_font_colors(self, fg_colors, bg_colors):
        self.fg_colors = np.array(fg_colors)
        self.bg_colors = np.array(bg_colors)

    def update_font_colors(self, fg_colors: np.ndarray, bg_colors: np.ndarray):
        nlines = len(self)
        if nlines > 0:
            self.fg_colors += fg_colors / nlines
            self.bg_colors += bg_colors / nlines

    def get_font_colors(self, bgr=False):

        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)

        if bgr:
            frgb = frgb[::-1]
            brgb = brgb[::-1]

        if self.adjust_bg_color:
            fg_avg = np.mean(frgb)
            if color_difference(frgb, brgb) < 30:
                brgb = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)

        return frgb, brgb

    @property
    def direction(self):
        """Render direction determined through used language or aspect ratio."""
        if self._direction not in ('h', 'v', 'hr', 'vr'):
            d = LANGUAGE_ORIENTATION_PRESETS.get(self.target_lang)
            if d in ('h', 'v', 'hr', 'vr'):
                return d

            if self.aspect_ratio < 1:
                return 'v'
            else:
                return 'h'
        return self._direction

    @property
    def vertical(self):
        return self.direction.startswith('v')

    @property
    def horizontal(self):
        return self.direction.startswith('h')

    @property
    def alignment(self):
        """Render alignment(/gravity) determined through used language."""
        if self._alignment in ('left', 'center', 'right'):
            return self._alignment
        if len(self.lines) == 1:
            return 'center'

        if self.direction == 'h':
            return 'center'
        elif self.direction == 'hr':
            return 'right'
        else:
            return 'left'

        # x1, y1, x2, y2 = self.xyxy
        # polygons = self.unrotated_polygons
        # polygons = polygons.reshape(-1, 4, 2)
        # print(self.polygon_aspect_ratio, self.xyxy)
        # print(polygons[:, :, 0] - x1)
        # print()
        # if self.polygon_aspect_ratio < 1:
        #     left_std = abs(np.std(polygons[:, :2, 1] - y1))
        #     right_std = abs(np.std(polygons[:, 2:, 1] - y2))
        #     center_std = abs(np.std(((polygons[:, :, 1] + polygons[:, :, 1]) - (y2 - y1)) / 2))
        #     print(center_std)
        #     print('a', left_std, right_std, center_std)
        # else:
        #     left_std = abs(np.std(polygons[:, ::2, 0] - x1))
        #     right_std = abs(np.std(polygons[:, 2:, 0] - x2))
        #     center_std = abs(np.std(((polygons[:, :, 0] + polygons[:, :, 0]) - (x2 - x1)) / 2))
        # min_std = min(left_std, right_std, center_std)
        # if left_std == min_std:
        #     return 'left'
        # elif right_std == min_std:
        #     return 'right'
        # else:
        #     return 'center'

    @property
    def stroke_width(self):
        diff = color_difference(*self.get_font_colors())
        if diff > 15:
            return self.default_stroke_width
        return 0


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)

    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated


def sort_regions(regions: List[TextBlock], right_to_left=True) -> List[TextBlock]:
    # Sort regions from right to left, top to bottom
    sorted_regions = []
    for region in sorted(regions, key=lambda region: region.center[1]):
        for i, sorted_region in enumerate(sorted_regions):
            if region.center[1] > sorted_region.xyxy[3]:
                continue
            if region.center[1] < sorted_region.xyxy[1]:
                sorted_regions.insert(i + 1, region)
                break

            # y center of region inside sorted_region so sort by x instead
            if right_to_left and region.center[0] > sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center[0] < sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
        else:
            sorted_regions.append(region)
    return sorted_regions


# def sort_textblk_list(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
#     if len(blk_list) == 0:
#         return blk_list
#     num_ja = 0
#     xyxy = []
#     for blk in blk_list:
#         if blk.language == 'ja':
#             num_ja += 1
#         xyxy.append(blk.xyxy)
#     xyxy = np.array(xyxy)
#     flip_lr = num_ja > len(blk_list) / 2
#     im_oriw = im_w
#     if im_w > im_h:
#         im_w /= 2
#     num_gridy, num_gridx = 4, 3
#     img_area = im_h * im_w
#     center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
#     if flip_lr:
#         if im_w != im_oriw:
#             center_x = im_oriw - center_x
#         else:
#             center_x = im_w - center_x
#     grid_x = (center_x / im_w * num_gridx).astype(np.int32)
#     center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
#     grid_y = (center_y / im_h * num_gridy).astype(np.int32)
#     grid_indices = grid_y * num_gridx + grid_x
#     grid_weights = grid_indices * img_area + 1.2 * (center_x - grid_x * im_w / num_gridx) + (center_y - grid_y * im_h / num_gridy)
#     if im_w != im_oriw:
#         grid_weights[np.where(grid_x >= num_gridx)] += img_area * num_gridy * num_gridx

#     for blk, weight in zip(blk_list, grid_weights):
#         blk.sort_weight = weight
#     blk_list.sort(key=lambda blk: blk.sort_weight)
#     return blk_list

# # TODO: Make these cached_properties
# def examine_textblk(blk: TextBlock, im_w: int, im_h: int, sort: bool = False) -> None:
#     lines = blk.lines_array()
#     middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
#     vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors of textlines
#     vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors of textlines
#     # if sum of vertical vectors is longer, then text orientation is vertical, and vice versa.
#     center_pnts = (lines[:, 0] + lines[:, 2]) / 2
#     v = np.sum(vec_v, axis=0)
#     h = np.sum(vec_h, axis=0)
#     norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
#     if blk.language == 'ja':
#         vertical = norm_v > norm_h
#     else:
#         vertical = norm_v > norm_h * 2
#     # calculate distance between textlines and origin 
#     if vertical:
#         primary_vec, primary_norm = v, norm_v
#         distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
#         font_size = int(round(norm_h / len(lines)))
#     else:
#         primary_vec, primary_norm = h, norm_h
#         distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
#         font_size = int(round(norm_v / len(lines)))

#     rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)     # rotation angle of textlines
#     distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
#     rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
#     distance = np.abs(np.sin(rad_matrix) * distance)
#     blk.lines = lines.astype(np.int32).tolist()
#     blk.distance = distance
#     blk.angle = rotation_angle
#     if vertical:
#         blk.angle -= 90
#     if abs(blk.angle) < 3:
#         blk.angle = 0
#     blk.font_size = font_size
#     blk.vertical = vertical
#     blk.vec = primary_vec
#     blk.norm = primary_norm
#     if sort:
#         blk.sort_lines()

# def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.4, distance_tol=2) -> bool:
#     if blk2.merged:
#         return False
#     fntsize_div = blk.font_size / blk2.font_size
#     num_l1, num_l2 = len(blk), len(blk2)
#     fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
#     vec_prod = blk.vec @ blk2.vec
#     vec_sum = blk.vec + blk2.vec
#     cos_vec = vec_prod / blk.norm / blk2.norm
#     distance = blk2.distance[-1] - blk.distance[-1]
#     distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
#     l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
#     if not l1.intersects(l2):
#         if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
#             return False
#         if abs(cos_vec) < 0.866:   # cos30
#             return False
#         # if distance > distance_tol * fntsz_avg or distance_p1 > fntsz_avg * 2.5:
#         if distance > distance_tol * fntsz_avg:
#             return False
#         if blk.vertical and blk2.vertical and distance_p1 > fntsz_avg * 2.5:
#             return False
#     # merge
#     blk.lines.append(blk2.lines[0])
#     blk.vec = vec_sum
#     blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
#     if blk.vertical:
#         blk.angle -= 90
#     blk.norm = np.linalg.norm(vec_sum)
#     blk.distance = np.append(blk.distance, blk2.distance[-1])
#     blk.font_size = fntsz_avg
#     blk2.merged = True
#     return True

# def merge_textlines(blk_list: List[TextBlock]) -> List[TextBlock]:
#     if len(blk_list) < 2:
#         return blk_list
#     blk_list.sort(key=lambda blk: blk.distance[0])
#     merged_list = []
#     for ii, current_blk in enumerate(blk_list):
#         if current_blk.merged:
#             continue
#         for jj, blk in enumerate(blk_list[ii+1:]):
#             try_merge_textline(current_blk, blk)
#         merged_list.append(current_blk)
#     for blk in merged_list:
#         blk.adjust_bbox(with_bbox=False)
#     return merged_list

# def split_textblk(blk: TextBlock):
#     font_size, distance, lines = blk.font_size, blk.distance, blk.lines
#     l0 = np.array(blk.lines[0])
#     lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
#     distance_tol = font_size * 2
#     current_blk = copy.deepcopy(blk)
#     current_blk.lines = [l0]
#     sub_blk_list = [current_blk]
#     textblock_splitted = False
#     for jj, line in enumerate(lines[1:]):
#         l1, l2 = Polygon(lines[jj]), Polygon(line)
#         split = False
#         if not l1.intersects(l2):
#             line_disance = abs(distance[jj+1] - distance[jj])
#             if line_disance > distance_tol:
#                 split = True
#             elif blk.vertical and abs(blk.angle) < 15:
#                 if len(current_blk.lines) > 1 or line_disance > font_size:
#                     split = abs(lines[jj][0][1] - line[0][1]) > font_size
#         if split:
#             current_blk = copy.deepcopy(current_blk)
#             current_blk.lines = [line]
#             sub_blk_list.append(current_blk)
#         else:
#             current_blk.lines.append(line)
#     if len(sub_blk_list) > 1:
#         textblock_splitted = True
#         for current_blk in sub_blk_list:
#             current_blk.adjust_bbox(with_bbox=False)
#     return textblock_splitted, sub_blk_list

# def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True) -> List[TextBlock]:
#     blk_list: List[TextBlock] = []
#     scattered_lines = {'ver': [], 'hor': []}
#     for bbox, lang_id, conf in zip(*blks):
#         # cls could give wrong result
#         blk_list.append(TextBlock(bbox, language=LANG_LIST[lang_id]))

#     # step1: filter & assign lines to textblocks
#     bbox_score_thresh = 0.4
#     mask_score_thresh = 0.1
#     for line in lines:
#         bx1, bx2 = line[:, 0].min(), line[:, 0].max()
#         by1, by2 = line[:, 1].min(), line[:, 1].max()
#         bbox_score, bbox_idx = -1, -1
#         line_area = (by2-by1) * (bx2-bx1)
#         for i, blk in enumerate(blk_list):
#             score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
#             if bbox_score < score:
#                 bbox_score = score
#                 bbox_idx = i
#         if bbox_score > bbox_score_thresh:
#             blk_list[bbox_idx].lines.append(line)
#         else: # if no textblock was assigned, check whether there is "enough" textmask
#             if mask is not None:
#                 mask_score = mask[by1: by2, bx1: bx2].mean() / 255
#                 if mask_score < mask_score_thresh:
#                     continue
#             blk = TextBlock([bx1, by1, bx2, by2], [line])
#             examine_textblk(blk, im_w, im_h, sort=False)
#             if blk.vertical:
#                 scattered_lines['ver'].append(blk)
#             else:
#                 scattered_lines['hor'].append(blk)

#     # step2: filter textblocks, sort & split textlines
#     final_blk_list = []
#     for blk in blk_list:
#         # filter textblocks 
#         if len(blk.lines) == 0:
#             bx1, by1, bx2, by2 = blk.xyxy
#             if mask is not None:
#                 mask_score = mask[by1: by2, bx1: bx2].mean() / 255
#                 if mask_score < mask_score_thresh:
#                     continue
#             xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
#             blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
#         examine_textblk(blk, im_w, im_h, sort=True)

#         # split manga text if there is a distance gap
#         textblock_splitted = False
#         if len(blk.lines) > 1:
#             if blk.language == 'ja':
#                 textblock_splitted = True
#             elif blk.vertical:
#                 textblock_splitted = True
#         if textblock_splitted:
#             textblock_splitted, sub_blk_list = split_textblk(blk)
#         else:
#             sub_blk_list = [blk]
#         # modify textblock to fit its textlines
#         if not textblock_splitted:
#             for blk in sub_blk_list:
#                 blk.adjust_bbox(with_bbox=True)
#         final_blk_list += sub_blk_list

#     # step3: merge scattered lines, sort textblocks by "grid"
#     final_blk_list += merge_textlines(scattered_lines['hor'])
#     final_blk_list += merge_textlines(scattered_lines['ver'])
#     if sort_blklist:
#         final_blk_list = sort_textblk_list(final_blk_list, im_w, im_h)

#     for blk in final_blk_list:
#         if blk.language != 'ja' and not blk.vertical:
#             num_lines = len(blk.lines)
#             if num_lines == 0:
#                 continue
#             # blk.line_spacing = blk.bounding_rect()[3] / num_lines / blk.font_size
#             expand_size = max(int(blk.font_size * 0.1), 3)
#             rad = np.deg2rad(blk.angle)
#             shifted_vec = np.array([[[-1, -1],[1, -1],[1, 1],[-1, 1]]])
#             shifted_vec = shifted_vec * np.array([[[np.sin(rad), np.cos(rad)]]]) * expand_size
#             lines = blk.lines_array() + shifted_vec
#             lines[..., 0] = np.clip(lines[..., 0], 0, im_w-1)
#             lines[..., 1] = np.clip(lines[..., 1], 0, im_h-1)
#             blk.lines = lines.astype(np.int64).tolist()
#             blk.font_size += expand_size

#     return final_blk_list

def visualize_textblocks(canvas, blk_list: List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for i, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        for j, line in enumerate(blk.lines):
            cv2.putText(canvas, str(j), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        cv2.polylines(canvas, [blk.min_rect], True, (127,127,0), 2)
        cv2.putText(canvas, str(i), (bx1, by1 + lw), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, 'a: %.2f' % blk.angle, [bx1, center[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, 'x: %s' % bx1, [bx1, center[1] + 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, 'y: %s' % by1, [bx1, center[1] + 60], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
    return canvas
