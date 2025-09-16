from typing import List, Tuple

import cv2
import numpy as np

from .panel import get_panels_from_array
from .textblock import TextBlock


def sort_regions(
        regions: List[TextBlock],
        right_to_left: bool = True,
        img: np.ndarray = None,
        force_simple_sort: bool = False
) -> List[TextBlock]:
    if not regions:
        return []

    # If simple sort is forced, use it and return immediately.
    if force_simple_sort:
        return _simple_sort(regions, right_to_left)

    # 1. Panel detection + sorting within panels
    if img is not None:
        try:
            panels_raw = get_panels_from_array(img, rtl=right_to_left)
            # Convert to [x1, y1, x2, y2]
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
            # Use the customised sorter that keeps vertically stacked panels together.
            panels = _sort_panels_fill(panels, right_to_left)

            # Assign panel_index to each region
            for r in regions:
                cx, cy = r.center
                r.panel_index = -1
                for idx, (x1, y1, x2, y2) in enumerate(panels):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        r.panel_index = idx
                        break
                if r.panel_index < 0:
                    # If not inside any panel, find the closest one
                    dists = [
                        ((max(x1 - cx, 0, cx - x2)) ** 2 + (max(y1 - cy, 0, cy - y2)) ** 2, i)
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
                panel_sorted = sort_regions(grouped[pi], right_to_left, img=None, force_simple_sort=False)
                sorted_all += panel_sorted
            return sorted_all

        except (cv2.error, MemoryError, Exception) as e:
            # When panel detection fails, use simple sort, log a warning but continue translation.
            from ..utils import get_logger
            logger = get_logger('textblock')
            logger.warning(
                f'Panel detection failed ({e.__class__.__name__}: {str(e)[:100]}), using simple text sorting')

            # Use the simple sorting logic as a fallback.
            return _simple_sort(regions, right_to_left)

    # 2. Smart sorting (if img is None and not forced simple)
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
    """Return panels in desired reading order.

    1. Panels are processed row-by-row from top to bottom (smallest *y1* first).
    2. Inside a row we proceed from right to left (*x1* descending when RTL, ascending otherwise).
    3. **Key point**: when moving horizontally, every panel that *roughly* shares
       the same x-range (both *x1* and *x2* close) with the current one is treated
       as a vertical stack and is inserted immediately after the current panel
       (top-to-bottom).  This mirrors how humans read stacked panels.

    The logic assumes panels fill the whole page (common in manga pages).
    """

    if not panels:
        return panels

    # Make a working copy we can pop from.
    remaining = sorted(list(panels), key=lambda p: p[1])  # sort by top-y first
    ordered: List[Tuple[int, int, int, int]] = []

    # Dynamic thresholds based on average panel size for reasonable robustness.
    avg_w = np.mean([p[2] - p[0] for p in remaining])
    avg_h = np.mean([p[3] - p[1] for p in remaining])
    X_THR = max(10, avg_w * 0.1)  # panels whose x-range differs <10% width are considered same column
    Y_THR = max(10, avg_h * 0.3)  # y-difference to decide panels are on the same row

    while remaining:
        # Start a new row from the current top-most panel
        base_y = remaining[0][1]

        # Gather all panels whose top-y 距离 base_y 不超过阈值 → 同一行
        row = []
        i = 0
        while i < len(remaining):
            if abs(remaining[i][1] - base_y) <= Y_THR:
                row.append(remaining.pop(i))
            else:
                i += 1

        # Sort that row right-to-left (或 LTR) 再加入
        row.sort(key=lambda p: (-p[0] if right_to_left else p[0]))
        ordered.extend(row)

    return ordered


def visualize_textblocks(canvas: np.ndarray, blk_list: List[TextBlock], show_panels: bool = False,
                         img_rgb: np.ndarray = None, right_to_left: bool = True):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width

    # Panel detection and drawing
    panels = None
    if show_panels and img_rgb is not None:
        try:
            panels_raw = get_panels_from_array(img_rgb, rtl=right_to_left)
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
            # Use the customised sorter that keeps vertically stacked panels together.
            panels = _sort_panels_fill(panels, right_to_left)

            # Draw panel boxes and order
            for panel_idx, (x1, y1, x2, y2) in enumerate(panels):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), lw)  # Magenta color for panels
                # Put panel number inside the box with deep blue color for better visibility and aesthetics
                cv2.putText(canvas, str(panel_idx), (x1 + 5, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX,
                            lw / 2, (200, 100, 0), max(lw - 1, 1), cv2.LINE_AA)
        except Exception as e:
            from ..utils import get_logger
            logger = get_logger('textblock')
            logger.warning(f'Panel visualization failed: {e}')

    for i, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        for j, line in enumerate(blk.lines):
            cv2.putText(canvas, str(j), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 127, 0), 1)
            cv2.polylines(canvas, [line], True, (0, 127, 255), 2)
        cv2.polylines(canvas, [blk.min_rect], True, (127, 127, 0), 2)
        cv2.putText(canvas, str(i), (bx1, by1 + lw), 0, lw / 3, (255, 127, 127), max(lw - 1, 1), cv2.LINE_AA)
        center = [int((bx1 + bx2) / 2), int((by1 + by2) / 2)]

        angle_text = 'a: %.2f' % blk.angle
        x_text = 'x: %s' % bx1
        y_text = 'y: %s' % by1

        # 添加描边效果，文本居中
        def put_text_with_outline(text, center_x, y, font_size=0.8, thickness=2, color=(127, 127, 255)):

            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
            text_x = center_x - text_width // 2

            # 绘制描边
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                cv2.putText(canvas, text, (text_x + dx, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (35, 24, 22), thickness)
            # 绘制原始颜色的主文本
            cv2.putText(canvas, text, (text_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)

        # 在文本框水平中央位置绘制带描边的文本
        center_x = center[0]
        put_text_with_outline(angle_text, center_x, center[1] - 10)
        put_text_with_outline(x_text, center_x, center[1] + 15)
        put_text_with_outline(y_text, center_x, center[1] + 40)
    return canvas
