import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPoint
from functools import cached_property
import copy
import re
import py3langid as langid
from .panel import get_panels_from_array
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
    'KOR': 'h',
    'PLK': 'h',
    'PTB': 'h',
    'ROM': 'h',
    'RUS': 'h',
    'ESP': 'h',
    'TRK': 'h',
    'UKR': 'h',
    'VIN': 'h',
    'ARA': 'hr', # horizontal reversed (right to left)
    'FIL': 'h'
}

class TextBlock(object):
    """
    Object that stores a block of text made up of textlines.
    """
    def __init__(self, lines: List[Tuple[int, int, int, int]],
                 texts: List[str] = None,
                 language: str = 'unknown',
                 font_size: float = -1,
                 angle: float = 0,
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
        if self.text and len(texts) > 1:
            for txt in texts[1:]:
                first_cjk = '\u3000' <= self.text[-1] <= '\u9fff'
                second_cjk = txt and ('\u3000' <= txt[0] <= '\u9fff')
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
        im_h, im_w = img.shape[:2]

        line = np.round(np.array(self.lines[line_idx])).astype(np.int64)

        x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].min(), line[:, 0].max(), line[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_croped = img[y1: y2, x1: x2]
        
        direction = 'v' if self.src_is_vertical else 'h'

        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        norm_v = np.linalg.norm(vec_v)
        norm_h = np.linalg.norm(vec_h)

        if textheight is None:
            if direction == 'h' :
                textheight = int(norm_v)
            else:
                textheight = int(norm_h)
        
        if norm_v <= 0 or norm_h <= 0:
            print('invalid textpolygon to target img')
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == 'h' :
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
        elif direction == 'v' :
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
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

            # 根据region中面积最大的文本框的宽高比来判断排版方向
            if len(self.lines) > 0:
                # 计算每个检测框的面积和宽高比
                max_area = 0
                largest_box_aspect_ratio = 1
                
                for line in self.lines:
                    # 计算检测框的面积
                    line_polygon = Polygon(line)
                    area = line_polygon.area
                    
                    if area > max_area:
                        max_area = area
                        # 计算该检测框的宽高比
                        # 获取检测框的边界框
                        x_coords = line[:, 0]
                        y_coords = line[:, 1]
                        width = np.max(x_coords) - np.min(x_coords)
                        height = np.max(y_coords) - np.min(y_coords)
                        largest_box_aspect_ratio = width / height if height > 0 else 1
                
                # 根据面积最大的检测框的宽高比判断方向
                if largest_box_aspect_ratio < 1:
                    return 'v'
                else:
                    return 'h'
            else:
                # 如果没有lines，则使用整体的宽高比作为fallback
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


def sort_regions(
    regions: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None
) -> List[TextBlock]:
    
    if not regions:
        return []

    # 1. 分镜检测 + 分镜内排序
    if img is not None:
        try:
            panels_raw = get_panels_from_array(img, rtl=right_to_left)
            # 转 [x1,y1,x2,y2]
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
            # 对 panels 本身排序：先 y 再 x（RTL x 降序）
            panels.sort(key=lambda p: (p[1], -p[0] if right_to_left else p[0]))

            # 标记 panel_index
            for r in regions:
                cx, cy = r.center
                r.panel_index = -1
                for idx, (x1, y1, x2, y2) in enumerate(panels):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        r.panel_index = idx
                        break
                if r.panel_index < 0:
                    # 如果不在任何 panel 内，找最近的
                    dists = [
                        ((max(x1-cx,0,cx-x2))**2 + (max(y1-cy,0,cy-y2))**2, i)
                        for i,(x1,y1,x2,y2) in enumerate(panels)
                    ]
                    r.panel_index = min(dists)[1]

            # 按 panel_index 分组，然后递归调用 sort_regions（不传 img 用坐标排序）
            grouped = {}
            for r in regions:
                grouped.setdefault(r.panel_index, []).append(r)
            
            sorted_all = []
            for pi in sorted(grouped):
                panel_sorted = sort_regions(grouped[pi], right_to_left)
                sorted_all += panel_sorted
            return sorted_all
            
        except (cv2.error, MemoryError, Exception) as e:
            # 面板检测失败时使用简化排序，记录警告但不阻止翻译继续
            from ..utils import get_logger
            logger = get_logger('textblock')
            logger.warning(f'Panel detection failed ({e.__class__.__name__}: {str(e)[:100]}), using simple text sorting')
            
            # 使用简化排序逻辑并直接返回
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

    # 2. 智能排序
    xs = [r.center[0] for r in regions]
    ys = [r.center[1] for r in regions]
    
    # 改进的分散度计算：使用标准差
    if len(regions) > 1:
        x_std = np.std(xs) if len(xs) > 1 else 0
        y_std = np.std(ys) if len(ys) > 1 else 0
        
        # 使用标准差比值来判断排列方向
        is_horizontal = x_std > y_std
    else:
        # 只有一个文本块时，默认为纵向
        is_horizontal = False

    sorted_regions = []
    if is_horizontal:
        # 横向更分散：先 x 再 y
        primary = sorted(regions, key=lambda r: -r.center[0] if right_to_left else r.center[0])
        group, prev = [], None
        for r in primary:
            cx = r.center[0]
            if prev is not None and abs(cx - prev) > 20:
                group.sort(key=lambda r: r.center[1])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cx
        if group:
            group.sort(key=lambda r: r.center[1])
            sorted_regions += group
    else:
        # 纵向更分散：先 y 再 x
        primary = sorted(regions, key=lambda r: r.center[1])
        group, prev = [], None
        for r in primary:
            cy = r.center[1]
            if prev is not None and abs(cy - prev) > 15:
                group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cy
        if group:
            group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
            sorted_regions += group
    
    return sorted_regions


def visualize_textblocks(canvas: np.ndarray, blk_list: List[TextBlock]):
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
        
        angle_text = 'a: %.2f' % blk.angle
        x_text = 'x: %s' % bx1
        y_text = 'y: %s' % by1
        
        # 添加描边效果，文本居中
        def put_text_with_outline(text, center_x, y, font_size=0.8, thickness=2, color=(127,127,255)):
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
            text_x = center_x - text_width // 2
            
            # 绘制描边
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
                cv2.putText(canvas, text, (text_x+dx, y+dy), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_size, (35,24,22), thickness)
            # 绘制原始颜色的主文本
            cv2.putText(canvas, text, (text_x, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        # 在文本框水平中央位置绘制带描边的文本
        center_x = center[0]  
        put_text_with_outline(angle_text, center_x, center[1] - 10)
        put_text_with_outline(x_text, center_x, center[1] + 15)
        put_text_with_outline(y_text, center_x, center[1] + 40)
    return canvas
