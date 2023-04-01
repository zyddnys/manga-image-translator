import os
import cv2
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('rendering')

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
    return fg, bg

# def dst_points_intersect(pts1, pts2):
#     poly1, poly2 = Polygon(pts1), Polygon(pts2)
#     return poly1.intersects(poly2)

# def rearrange_dst_points(img, text_regions, dst_points_list):
#     n_dst_points_list = []

#     # Step 1: Collect regions with aspect ratio that does not work with render direction
#     v_regions = []
#     h_regions = []
#     other_regions = []
#     for region in text_regions:
#         # If target language can be rendered in both orientations
#         if LANGAUGE_ORIENTATION_PRESETS.get(region.target_lang) == 'auto':
#             continue
#         if region.aspect_ratio < 1 and region.direction == 'h':
#             h_regions.append(region)
#         elif region.aspect_ratio > 1 and region.direction == 'v':
#             v_regions.append(region)
#         else:
#             other_regions.append(region)

#     # Step 2: Group collected regions if they share same text bubble/area
#     for d in ('v', 'h'):
#         for region in (v_regions if d == 'v' else h_regions):
#             dst_points = region.min_rect

#             # ballon_mask, xyxy = extract_ballon_region(img, [0, 0, img.shape[1], img.shape[0]], enlarge_ratio=1, verbose=True)
#             if img is not None:
#                 # Group regions if their dst_points intersect
#                 ...

#                 # If original text_direction is vertical, sort groups from right to left
#                 # Else, sort groups from top to bottom
#             n_dst_points_list.append(dst_points)

#     # Step 3: Resize groups and single regions to remove intersections
#     # distribute based on minimum width/height requirements otherwise higher area means more favor

#     # Step 4: Rearrange grouped regions

# def generate_region_placement(img, text_regions):
#     # Expand dst_points to ballon region
#     dst_points_list = []
#     for region in text_regions:
#         ballon_mask, searched_xyxy = extract_ballon_region(img, region.xywh, enlarge_ratio=1.2, verbose=True)
#         ballon_xywh = np.array(cv2.boundingRect(cv2.findNonZero(ballon_mask)))
#         ballon_xywh[:2] += searched_xyxy[:2]

#         print(ballon_xywh)
#         dst_points_list.append(region.min_rect)

#     # To prevent intersections, rearrange or resize regions

#     # Resize

# def shrink_polygon_edge_towards_point(poly_pts, side_pt_idxs, target_pt):
#     """
#     Moves two neighboring points of poly_pts towards target_pt.
#     The edge of those two points will be made to touch target_pt.
#     """

#     # 1 and 2 are neighboring points in poly_pts
#     i1, i2 = side_pt_idxs
#     i3 = (i1 - 1) % 4 if (i1 - 1) % 4 != i2 else (i1 + 1) % 4
#     i4 = (i2 - 1) % 4 if (i2 - 1) % 4 != i1 else (i2 + 1) % 4

#     side_pt1 = poly_pts[i1]
#     side_pt2 = poly_pts[i2]
#     side_pt3 = poly_pts[i3]
#     side_pt4 = poly_pts[i4]
#     print(i1, i2, i3, i4)
#     print(side_pt1, side_pt2)

#     # Calculate functions and their intersections which will be the new points
#     m_13 = (side_pt1[1] - side_pt3[1]) / (side_pt1[0] - side_pt3[0])
#     m_24 = (side_pt2[1] - side_pt4[1]) / (side_pt2[0] - side_pt4[0])
#     t_13 = side_pt1[1] - m_13 * side_pt1[0]
#     t_24 = side_pt2[1] - m_24 * side_pt2[0]
#     if side_pt1[0] - side_pt2[0] == 0:
#         x1 = target_pt[0]
#         x2 = target_pt[0]
#     else:
#         m = (side_pt1[1] - side_pt2[1]) / (side_pt1[0] - side_pt2[0])
#         t = target_pt[1] - m * target_pt[0]
#         x1 = (t_13 - t) / (m - m_13)
#         x2 = (t_24 - t) / (m - m_24)

#     y1 = m_13 * x1 + t_13
#     y2 = m_24 * x2 + t_24

#     poly_pts[i1] = (x1, y1)
#     poly_pts[i2] = (x2, y2)
#     return poly_pts

# def remove_intersections(points_list: np.ndarray):
#     """
#     Resizes points of polygons to remove intersections with all other polygons.
#     Expects 4 point per polygon that has (4, 2) shape.
#     Ignores polygons that are completely within one another.
#     """
#     polygons = [make_valid(Polygon(pts)) for pts in points_list]
#     for i, (poly_pts1, poly1) in enumerate(zip(points_list, polygons)):
#         for j, (poly_pts2, poly2) in enumerate(zip(points_list[i+1:], polygons[i+1:])):
#             print('COMPARE', poly1, poly2)
#             if poly1.intersects(poly2):
#                 intersection = poly1.intersection(poly2)
#                 intersection_centroid = intersection.centroid.coords[0]
#                 intersection_pts = np.array(mapping(intersection)['coordinates'][0][:-1])
#                 print(intersection_centroid, intersection_pts)
#                 print(len(intersection_pts))

#                 # Sort intersection_pts to corresponding polygon
#                 ipts1 = []
#                 ipts2 = []
#                 outside_polygon_pts = [] # Can be pts1 or pts2
#                 remaining_points = []
#                 for pt in intersection_pts:
#                     added = False
#                     print(poly_pts1, poly_pts2, pt)
#                     print(np.all(poly_pts1 == pt, axis=1))
#                     print(np.all(poly_pts2 == pt, axis=1))
#                     if np.any(np.all(poly_pts1 == pt, axis=1)):
#                         ipts1.append(pt)
#                         added = True
#                     if np.any(np.all(poly_pts2 == pt, axis=1)):
#                         ipts2.append(pt)
#                         added = True
#                     if not added:
#                         remaining_points.append(pt)

#                 assert not (len(ipts1) == 0 and len(ipts2) == 0)

#                 if len(ipts1) == 0:
#                     outside_polygon_pts = poly_pts1
#                 elif len(ipts2) == 0:
#                     outside_polygon_pts = poly_pts2

#                 ipts1 = np.array(ipts1)
#                 ipts2 = np.array(ipts2)
#                 remaining_points = np.array(remaining_points)

#                 for poly_pts, ipts in zip((poly_pts1, poly_pts2), (ipts1, ipts2)):
#                     print('LEN', len(ipts), ipts)
#                     if len(ipts) == 1:
#                         pt_idx = np.all(poly_pts == ipts[0], axis=1).nonzero()[0]
#                         poly_pts[pt_idx] = intersection_centroid
#                         print('NOW', poly_pts)
#                     elif len(ipts) == 2:
#                         # Move edge between the two points to the centroid
#                         i1 = np.all(poly_pts == ipts[0], axis=1).nonzero()[0][0]
#                         i2 = np.all(poly_pts == ipts[1], axis=1).nonzero()[0][0]
#                         shrink_polygon_edge_towards_point(poly_pts, (i1, i2), intersection_centroid)
#                         print('UPDATED', poly_pts)
#                     elif len(ipts) == 3:
#                         ...
#                     # else completely inside the other polygon

#                 print('OUTSIDE', len(outside_polygon_pts), outside_polygon_pts)
#                 if outside_polygon_pts is not None:
#                     print(remaining_points)
#                     if len(remaining_points) == 2:
#                         # Move edge that contains both of the remaining_points
#                         # Find side which contains remaining_points
#                         for i1, pt1 in enumerate(outside_polygon_pts):
#                             i2 = (i1 + 1) % 4
#                             pt2 = outside_polygon_pts[i2]
#                             line = LineString((pt1, pt2))
#                             print(pt1, pt2)
#                             print(line.within(Point(*remaining_points[0])), line.distance(Point(*remaining_points[0])), Point(*remaining_points[0]))
#                             if line.distance(Point(*remaining_points[0])) <= 0.01:
#                                 print('TOUCHES')
#                                 shrink_polygon_edge_towards_point(outside_polygon_pts, (i1, i2), intersection_centroid)
#                                 break

#     return points_list

def resize_regions_to_font_size(img: np.ndarray, text_regions: List[TextBlock], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):
    if font_size_minimum == -1:
        # Automatically determine font_size by image size
        font_size_minimum = min(img.shape[0], img.shape[1]) / 200

    dst_points_list = []
    for region in text_regions:
        # Correct the fontsize
        if min(len(region.get_text()), len(region.translation)) > 0:
            region.font_size *= max(min(len(region.get_text()) / len(region.translation), 1.2), 0.8)
        if region.font_size > 50:
            region.font_size = 0.8 * region.font_size
        elif region.font_size > 30:
            region.font_size -= 5
        region.font_size = int(region.font_size)

        # Modify the fontsize based on user arguments and rescale dst_points accordingly
        new_font_size = region.font_size
        if font_size_fixed is not None:
            new_font_size = font_size_fixed
        elif new_font_size < font_size_minimum:
            new_font_size = max(region.font_size, font_size_minimum)
        new_font_size += font_size_offset

        target_scale = new_font_size / region.font_size
        region.font_size = new_font_size
        if target_scale != 1:
            # print('Rescaling to', target_scale)
            poly = Polygon(region.unrotated_min_rect[0])
            poly = affinity.scale(poly, xfact=target_scale, yfact=target_scale)
            dst_points = np.array(poly.exterior.coords[:4])
            dst_points = rotate_polygons(region.center, dst_points.reshape(1, -1), -region.angle).reshape(-1, 4, 2)

            # Clip to img width and height
            dst_points[..., 0] = dst_points[..., 0].clip(0, img.shape[1])
            dst_points[..., 1] = dst_points[..., 1].clip(0, img.shape[0])

            dst_points = dst_points.reshape((-1, 4, 2))
        else:
            dst_points = region.min_rect

        dst_points_list.append(dst_points)
    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    render_mask: np.ndarray = None,
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # dst_points_list = []
    # if rearrange_regions:
    #     # Rearrange regions in same ballon region if render direction is different from source
    #     # dst_points_list = generate_region_placement(img, text_regions)

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # Remove intersections
    # TODO: Fix
    # remove_intersections([pts[0] for pts in dst_points_list])

    # Render text
    for region, dst_points in zip(text_regions, dst_points_list):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)

        logger.info(f'text: {region.get_text()}')
        logger.info(f' trans: {region.translation}')
        logger.info(f'font_size: {region.font_size}')

        img = render(img, region, dst_points, region.alignment)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    alignment: str,
):

    # -- Disabled upscaled rendering because it made it difficult to set a fixed font size
    # # Round font_size to fixed powers of 2, to take advantage of the font caching.
    # # Generated image snippet with text will be downscaled to dst_points after rendering.
    # # snippet_font_size = findNextPowerOf2(region.font_size) * text_mag_ratio
    # # enlarge_ratio = snippet_font_size / region.font_size
    # # while True:
    # #     enlarged_w = round(enlarge_ratio * region.unrotated_size[0])
    # #     enlarged_h = round(enlarge_ratio * region.unrotated_size[1])
    # #     cols = enlarged_w // (snippet_font_size * 1.3)
    # #     rows = enlarged_h // (snippet_font_size * 1.3)
    # #     if rows * cols < len(region.translation):
    # #         enlarge_ratio *= 1.1
    # #         continue
    # #     break

    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    if region.direction == 'h':
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.translation,
            round(norm_h[0]),
            alignment,
            fg,
            bg,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.translation,
            round(norm_v[0]),
            alignment,
            fg,
            bg,
        )
    h, w, _ = temp_box.shape
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    if r_temp > r_orig:
        h_ext = int(w / (2 * r_orig) - h / 2)
        box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
        box[h_ext:h + h_ext, 0:w] = temp_box
    else:
        w_ext = int((h * r_orig - w) / 2)
        box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        box[0:h, w_ext:w_ext+w] = temp_box
    #h_ext += region_ext
    #w_ext += region_ext

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = np.clip(cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 255)
    canvas_region = rgba_region[:, :, 0: 3]
    mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
    img = np.clip((img.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '') -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, size_tol=1.2, original_img=original_img, downscale_constraint=0.8)
