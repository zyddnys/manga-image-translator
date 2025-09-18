import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPoint
from functools import cached_property
import copy
import re
import py3langid as langid

from .generic import color_difference, is_right_to_left_char, is_valuable_char
from .log import get_logger
from ..panel_detection import dispatch as dispatch_panel_detection
from ..config import PanelDetector, PanelDetectorConfig

logger = get_logger('textblock')
# from ..detection.ctd_utils.utils.imgproc_utils import union_area, xywh2xyxypoly

logger = get_logger('textblock')

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
    'POL': 'h',
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
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.texts = texts if texts is not None else []
        self.text = texts[0] if texts else ""
        if self.text and len(texts) > 1:
            for txt in texts[1:]:
                first_cjk = '\u3000' <= self.text[-1] <= '\u9fff'
                second_cjk = txt and ('\u3000' <= txt[0] <= '\u9fff')
                if first_cjk or second_cjk:
                    self.text += txt
                else:
                    self.text += ' ' + txt
        self.prob = prob

        self.translation = translation

        self.fg_colors = fg_color
        self.bg_colors = bg_color

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
            if direction == 'h':
                textheight = int(norm_v)
            else:
                textheight = int(norm_h)
        
        if norm_v <= 0 or norm_h <= 0:
            print('invalid textpolygon to target img')
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == 'h':
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
        elif direction == 'v':
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

            # Determine layout direction based on aspect ratio of largest text box in region
            if len(self.lines) > 0:
                # Calculate area and aspect ratio for each detection box
                max_area = 0
                largest_box_aspect_ratio = 1
                
                for line in self.lines:
                    # Calculate detection box area
                    line_polygon = Polygon(line)
                    area = line_polygon.area
                    
                    if area > max_area:
                        max_area = area
                        # Calculate aspect ratio of this detection box
                        # Get bounding box of detection box                  
                        x_coords = line[:, 0]
                        y_coords = line[:, 1]
                        width = np.max(x_coords) - np.min(x_coords)
                        height = np.max(y_coords) - np.min(y_coords)
                        largest_box_aspect_ratio = width / height if height > 0 else 1
                
                # Determine direction based on aspect ratio of largest detection box
                if largest_box_aspect_ratio < 1:
                    return 'v'
                else:
                    return 'h'
            else:
                # If no lines, use overall aspect ratio as fallback
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
            if self.vertical:
                return 'left'
            else:
                return 'center'

        if self.direction == 'h':
            return 'center'
        elif self.direction == 'hr':
            return 'right'
        else:
            return 'left'

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


def _simple_sort(regions: List[TextBlock], right_to_left: bool) -> List[TextBlock]:
    """
    A simple fallback sorting logic. Sorts regions from top to bottom,
    then by x-coordinate based on reading direction.
    """
    sorted_regions = []
    # Sort primarily by the y-coordinate of the center
    for region in sorted(regions, key=lambda r: r.center[1]):
        for i, sorted_region in enumerate(sorted_regions):
            # If the current region is clearly below a sorted region, continue
            if region.center[1] > sorted_region.xyxy[3]:
                continue
            # If the current region is clearly above a sorted region, it means we went too far
            if region.center[1] < sorted_region.xyxy[1]:
                sorted_regions.insert(i, region)
                break

            # y-center of the region is within the y-range of the sorted_region, so sort by x instead
            if right_to_left and region.center[0] > sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center[0] < sorted_region.center[0]:
                sorted_regions.insert(i, region)
                break
        else:
            # If the loop finishes without breaking, append the region to the end
            sorted_regions.append(region)
    
    return sorted_regions
    
def _sort_panels_fill(panels: List[Tuple[int, int, int, int]], right_to_left: bool) -> List[Tuple[int, int, int, int]]:
    """Return panels in desired reading order with simplified vertical stack handling.

    Improved algorithm:
    1. Check for nested panels (if a panel contains 70%+ of another panel's area, it forms a single stack)
    2. Identify vertical stacks (panels with high x-overlap and vertical adjacency)
    3. Post-stack validation: check if remaining panels have 70%+ cumulative overlap with multiple stack members
    4. If validation fails, cancel the stack and isolate the base panel
    5. Treat each stack as a single unit for sorting
    6. Sort units by row (top y-coordinate) then by column (x-coordinate)
    7. Maintain top-to-bottom order within each vertical stack
    
    Nested panel handling:
    - Pre-check: Panels containing 70%+ of other panels' area are isolated as single-panel stacks
    - Post-check: Stacks are cancelled if remaining panels overlap 70%+ with multiple stack members
    - This prevents both container panels and complex overlapping scenarios from forming incorrect stacks
    - Ensures proper reading order for complex nested and overlapping layouts
    """

    if not panels:
        return panels

    # Dynamic thresholds based on average panel size
    avg_w = np.mean([p[2] - p[0] for p in panels])
    avg_h = np.mean([p[3] - p[1] for p in panels])
    nested_threshold = 0.7  # If 70%+ of another panel's area is within the base panel, consider it nested
    # Note: Y_THR is now calculated dynamically for each row based on current panel height

    # Step 1: Identify vertical stacks
    vertical_stacks = []
    remaining = list(panels)
    stack_id = 0

    while remaining:
        base_panel = remaining.pop(0)
        stack = [base_panel]
        base_x1, base_y1, base_x2, base_y2 = base_panel

        # Check if base panel contains most area of other panels (nested check)
        def calculate_overlap_ratio(panel_a, panel_b):
            """Calculate overlap ratio of panel_b within panel_a"""
            ax1, ay1, ax2, ay2 = panel_a
            bx1, by1, bx2, by2 = panel_b
            
            # Calculate overlap area
            overlap_x1 = max(ax1, bx1)
            overlap_y1 = max(ay1, by1)
            overlap_x2 = min(ax2, bx2)
            overlap_y2 = min(ay2, by2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                panel_b_area = (bx2 - bx1) * (by2 - by1)
                return overlap_area / panel_b_area if panel_b_area > 0 else 0
            return 0
        
        # Check if base panel contains most area of other panels
        contains_nested_panels = False
        
        for other_panel in remaining:
            overlap_ratio = calculate_overlap_ratio(base_panel, other_panel)
            if overlap_ratio >= nested_threshold:
                contains_nested_panels = True
                break
        
        if contains_nested_panels:
            # Sort stack top-to-bottom (though only one panel)
            stack.sort(key=lambda p: p[1])
            vertical_stacks.append(stack)
            stack_id += 1
            continue

        # Look for panels that vertically stack with base_panel
        i = 0
        while i < len(remaining):
            panel = remaining[i]
            x1, y1, x2, y2 = panel

            # Check x-range overlap
            x_overlap = min(base_x2, x2) - max(base_x1, x1)
            x_union = max(base_x2, x2) - min(base_x1, x1)

            # Check if vertically adjacent (allowing small gaps)
            vertical_gap = min(abs(y1 - base_y2), abs(base_y1 - y2))
            
            # Calculate overlap ratio
            overlap_ratio = x_overlap / x_union if x_union > 0 else 0
            gap_threshold = max(10, avg_w * 0.15)

            # 85%+ x-overlap and small vertical gap = vertical stack
            if (x_overlap > 0 and x_union > 0 and overlap_ratio > 0.85 and
                vertical_gap <= gap_threshold):
                stack.append(remaining.pop(i))
                # Update stack bounds
                base_x1 = min(base_x1, x1)
                base_x2 = max(base_x2, x2)
                base_y1 = min(base_y1, y1)
                base_y2 = max(base_y2, y2)
                i = 0  # Restart search for more stack members
            else:
                i += 1

        # Stack overlap check: if stack contains multiple panels, check if remaining panels overlap with multiple stack panels
        if len(stack) > 1:
            stack_should_break = False
            
            for remaining_panel in remaining:
                rx1, ry1, rx2, ry2 = remaining_panel
                remaining_area = (rx2 - rx1) * (ry2 - ry1)
                total_overlap_area = 0
                overlapping_count = 0
                
                # Calculate cumulative overlap area between remaining panel and all panels in stack
                for stack_panel in stack:
                    sx1, sy1, sx2, sy2 = stack_panel
                    
                    # Calculate overlap area
                    overlap_x1 = max(rx1, sx1)
                    overlap_y1 = max(ry1, sy1)
                    overlap_x2 = min(rx2, sx2)
                    overlap_y2 = min(ry2, sy2)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        total_overlap_area += overlap_area
                        overlapping_count += 1
                
                # Calculate cumulative overlap ratio
                overlap_ratio = total_overlap_area / remaining_area if remaining_area > 0 else 0
                
                # If cumulative overlap reaches threshold and involves multiple stack panels, cancel stacking
                if overlap_ratio >= nested_threshold and overlapping_count >= 2:
                    stack_should_break = True
                    break
            
            if stack_should_break:
                # Cancel stacking: return all panels except base panel back to remaining
                base_panel = stack[0]  # Base panel (first one)
                returned_panels = stack[1:]  # Other panels
                remaining.extend(returned_panels)
                stack = [base_panel]  # Keep only base panel

        # Sort stack top-to-bottom
        stack.sort(key=lambda p: p[1])
        vertical_stacks.append(stack)
        stack_id += 1

    # Step 2: Sort stacks using original row-based logic
    # Each stack is treated as a single unit with its top-left corner as reference
    stack_units = []
    for i, stack in enumerate(vertical_stacks):
        # Use the top-left corner of the stack as reference point
        top_panel = stack[0]  # Already sorted top-to-bottom
        stack_units.append((top_panel[0], top_panel[1], stack))  # (x, y, panels)

    # Apply original row-based sorting to stack units
    remaining_units = sorted(stack_units, key=lambda unit: unit[1])  # sort by top-y first
    ordered = []
    row_id = 0

    while remaining_units:
        # Start a new row from the current top-most unit
        base_unit = remaining_units[0]
        base_y = base_unit[1]
        
        # Calculate Y_THR based on current base unit's height
        base_stack = base_unit[2]  # Get the stack (list of panels)
        base_panel = base_stack[0]  # Get the top panel in the stack
        current_panel_height = base_panel[3] - base_panel[1]  # Calculate height
        Y_THR = max(10, current_panel_height * 0.15)  # y-difference threshold based on current panel

        # Gather all units whose top-y is within threshold of base_y
        row = []
        i = 0
        while i < len(remaining_units):
            unit_y = remaining_units[i][1]
            y_diff = abs(unit_y - base_y)
            
            if y_diff <= Y_THR:
                row.append(remaining_units.pop(i))
            else:
                i += 1

        # Sort that row right-to-left (or LTR) by x-coordinate
        row.sort(key=lambda unit: (-unit[0] if right_to_left else unit[0]))

        # Add all panels from each unit in the row
        for unit in row:
            unit_panels = unit[2]  # unit[2] is the list of panels in the stack
            for panel in unit_panels:
                ordered.append(panel)
        
        row_id += 1

    return ordered

async def sort_regions(
    regions: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None,
    device: str = 'cpu',
    panel_detector: str = 'none',
    panel_config: PanelDetectorConfig = None,
    ctx = None
) -> List[TextBlock]:

    if not regions:
        return []

    # If panel detection is disabled, use simple sort and return immediately.
    if panel_detector == 'none':
        return _simple_sort(regions, right_to_left)

    # 1. Panel detection + sorting within panels
    if img is not None:
        try:
            import time
            panel_start_time = time.time()

            # Convert string to enum
            if panel_detector == 'dl':
                detector_key = PanelDetector.dl
            elif panel_detector == 'kumiko':
                detector_key = PanelDetector.kumiko
            else:
                # Should not reach here due to early return for 'none'
                return _simple_sort(regions, right_to_left)

            logger.debug(f'Starting panel detection with {panel_detector} on {device}')
            panels_raw = await dispatch_panel_detection(detector_key, img, rtl=right_to_left, device=device, config=panel_config)
            panel_end_time = time.time()
            logger.debug(f'Panel detection completed in {panel_end_time - panel_start_time:.2f}s, found {len(panels_raw)} panels')
            # Convert to [x1, y1, x2, y2]
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]

            # Apply sorting based on detector type
            if detector_key == PanelDetector.kumiko:
                # Kumiko already sorts panels internally, trust its ordering
                pass
            else:
                # Use our custom sorter for other detectors (like DL)
                panels = _sort_panels_fill(panels, right_to_left)

            # Cache panel data in Context for reuse by visualize_textblocks
            if ctx is not None:
                ctx.panels_data = panels
                ctx.panel_detector_used = panel_detector
                ctx.panel_config_used = panel_config

            # Improved text assignment logic: select smallest containing panel
            for r in regions:
                cx, cy = r.center
                r.panel_index = -1
                containing_panels = []
                
                # Find all panels that contain this text block
                for idx, (x1, y1, x2, y2) in enumerate(panels):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        area = (x2 - x1) * (y2 - y1)  # Calculate panel area
                        containing_panels.append((area, idx))
                
                if containing_panels:
                    # Select smallest area panel (innermost in nested structure)
                    r.panel_index = min(containing_panels)[1]
                else:
                    # If not inside any panel, find the closest one
                    dists = [
                        ((max(x1-cx, 0, cx-x2))**2 + (max(y1-cy, 0, cy-y2))**2, i)
                        for i, (x1, y1, x2, y2) in enumerate(panels)
                    ]
                    if dists:
                        r.panel_index = min(dists)[1]

            # Group by panel_index, then recursively call sort_regions (without img for coordinate sorting)
            grouped = {}
            for r in regions:
                grouped.setdefault(r.panel_index, []).append(r)
            
            sorted_all = []
            # Ensure panels that couldn't be assigned are handled (e.g., panel_index=-1)
            # and sorted based on their panel index.
            for pi in sorted(grouped.keys()):
                panel_sorted = await sort_regions(grouped[pi], right_to_left, img=None, panel_detector='none')
                sorted_all += panel_sorted
            return sorted_all

        except (cv2.error, MemoryError, Exception) as e:
            # When panel detection fails, use simple sort as fallback
            return _simple_sort(regions, right_to_left)

    # 2. Smart sorting (if img is None and not forced simple)
    xs = [r.center[0] for r in regions]
    ys = [r.center[1] for r in regions]
    
    # Improved variance calculation: using standard deviation
    if len(regions) > 1:
        x_std = np.std(xs) if len(xs) > 1 else 0
        y_std = np.std(ys) if len(ys) > 1 else 0
        
        # Use standard deviation ratio to determine arrangement direction
        is_horizontal = x_std > y_std
    else:
        # When only one text block, default to vertical
        is_horizontal = False

    sorted_regions = []
    if is_horizontal:
        # More horizontal spread: sort by x first, then y
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
        # More vertical spread: sort by y first, then x
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


async def visualize_textblocks(canvas: np.ndarray, blk_list: List[TextBlock], show_panels: bool = False, img_rgb: np.ndarray = None, right_to_left: bool = True, device: str = 'cpu', panel_detector: str = 'dl', panel_config: PanelDetectorConfig = None, ctx = None):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width

    # Panel detection and drawing
    panels = None
    if show_panels and img_rgb is not None:
        try:
            # Try to use cached panel data first
            if (ctx is not None and
                hasattr(ctx, 'panels_data') and
                hasattr(ctx, 'panel_detector_used') and
                hasattr(ctx, 'panel_config_used') and
                ctx.panel_detector_used == panel_detector and
                ctx.panel_config_used == panel_config and
                ctx.panels_data is not None):
                # Use cached panel data from sort_regions
                panels = ctx.panels_data
            else:
                # Fallback: perform panel detection
                if panel_detector == 'none':
                    panels = []  # No panels when detection is disabled
                else:
                    if panel_detector == 'dl':
                        detector_key = PanelDetector.dl
                    elif panel_detector == 'kumiko':
                        detector_key = PanelDetector.kumiko
                    else:
                        panels = []  # Fallback for unknown detector
                        detector_key = None

                    if detector_key:
                        panels_raw = await dispatch_panel_detection(detector_key, img_rgb, rtl=right_to_left, device=device, config=panel_config)
                        panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]

                # Apply sorting based on detector type
                if detector_key == PanelDetector.kumiko:
                    # Kumiko already sorts panels internally, trust its ordering
                    pass
                else:
                    # Use our custom sorter for other detectors (like DL)
                    panels = _sort_panels_fill(panels, right_to_left)
            
            # Draw panel boxes and order
            for panel_idx, (x1, y1, x2, y2) in enumerate(panels):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), lw)  # Magenta color for panels
                # Put panel number inside the box with deep blue color for better visibility and aesthetics
                cv2.putText(canvas, str(panel_idx), (x1+5, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 
                           lw/2, (200, 100, 0), max(lw-1, 1), cv2.LINE_AA)
        except Exception as e:
            # Panel visualization failed, skip panel drawing
            logger.error(f"Panel visualization failed: {e}")
    
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
        
        # Add outline effect, text centered
        def put_text_with_outline(text, center_x, y, font_size=0.8, thickness=2, color=(127,127,255)):
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
            text_x = center_x - text_width // 2
            
            # Draw outline
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
                cv2.putText(canvas, text, (text_x+dx, y+dy), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_size, (35,24,22), thickness)
            # Draw main text in original color
            cv2.putText(canvas, text, (text_x, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
        
        # Draw outlined text at horizontal center of text box
        center_x = center[0]  
        put_text_with_outline(angle_text, center_x, center[1] - 10)
        put_text_with_outline(x_text, center_x, center[1] + 15)
        put_text_with_outline(y_text, center_x, center[1] + 40)
    return canvas