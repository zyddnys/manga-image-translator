import itertools
import numpy as np
from typing import List, Set
from collections import Counter
import networkx as nx
from shapely.geometry import Polygon

from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region

def split_text_region(
        bboxes: List[Quadrilateral],
        connected_region_indices: Set[int],
        width,
        height,
        gamma = 0.5,
        sigma = 2
    ) -> List[Set[int]]:

    connected_region_indices = list(connected_region_indices)

    # case 1
    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]

    # case 2
    if len(connected_region_indices) == 2:
        fs1 = bboxes[connected_region_indices[0]].font_size
        fs2 = bboxes[connected_region_indices[1]].font_size
        fs = max(fs1, fs2)

        # print(bboxes[connected_region_indices[0]].pts, bboxes[connected_region_indices[1]].pts)
        # print(fs, bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]), (1 + gamma) * fs)
        # print(bboxes[connected_region_indices[0]].angle, bboxes[connected_region_indices[1]].angle, 4 * np.pi / 180)

        if bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]) < (1 + gamma) * fs \
                and abs(bboxes[connected_region_indices[0]].angle - bboxes[connected_region_indices[1]].angle) < 0.2 * np.pi:
            return [set(connected_region_indices)]
        else:
            return [set([connected_region_indices[0]]), set([connected_region_indices[1]])]

    # case 3
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for (u, v) in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    # Get distances from neighbouring bboxes
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm='kruskal', data=True)
    edges = sorted(edges, key=lambda a: a[2]['weight'], reverse=True)
    distances_sorted = [a[2]['weight'] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = Polygon(b1.pts).distance(Polygon(b2.pts))
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    # print(edges)
    # print(f'std: {distances_std} < thrshold: {std_threshold}, mean: {distances_mean}')
    # print(f'{distances_sorted[0]} <= {distances_mean + distances_std * sigma}' \
    #         f' or {distances_sorted[0]} <= {fontsize * (1 + gamma)}' \
    #         f' or {distances_sorted[0] - distances_sorted[1]} < {distances_std * sigma}')

    if (distances_sorted[0] <= distances_mean + distances_std * sigma \
            or distances_sorted[0] <= fontsize * (1 + gamma)) \
            and (distances_std < std_threshold \
            or max_poly_distance == 0 and max_centroid_alignment < 5):
        return [set(connected_region_indices)]
    else:
        # (split_u, split_v, _) = edges[0]
        # print(f'split between "{bboxes[split_u].pts}", "{bboxes[split_v].pts}"')
        G = nx.Graph()
        for idx in connected_region_indices:
            G.add_node(idx)
        # Split out the most deviating bbox
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(split_text_region(bboxes, node_set, width, height))
        return ans

# def get_mini_boxes(contour):
#     bounding_box = cv2.minAreaRect(contour)
#     points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

#     index_1, index_2, index_3, index_4 = 0, 1, 2, 3
#     if points[1][1] > points[0][1]:
#         index_1 = 0
#         index_4 = 1
#     else:
#         index_1 = 1
#         index_4 = 0
#     if points[3][1] > points[2][1]:
#         index_2 = 2
#         index_3 = 3
#     else:
#         index_2 = 3
#         index_3 = 2

#     box = [points[index_1], points[index_2], points[index_3], points[index_4]]
#     box = np.array(box)
#     startidx = box.sum(axis=1).argmin()
#     box = np.roll(box, 4 - startidx, 0)
#     box = np.array(box)
#     return box

def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height):
    # step 0: merge quadrilaterals that belong to the same textline
    # u = 0
    # removed_counter = 0
    # while u < len(bboxes) - 1 - removed_counter:
    #     v = u
    #     while v < len(bboxes) - removed_counter:
    #         if quadrilateral_can_merge_region(bboxes[u], bboxes[v], aspect_ratio_tol=1.1, font_size_ratio_tol=1,
    #                                         char_gap_tolerance=1, char_gap_tolerance2=3, discard_connection_gap=0) \
    #            and abs(bboxes[u].centroid[0] - bboxes[v].centroid[0]) < 5 or abs(bboxes[u].centroid[1] - bboxes[v].centroid[1]) < 5:
    #                 bboxes[u] = merge_quadrilaterals(bboxes[u], bboxes[v])
    #                 removed_counter += 1
    #                 bboxes.pop(v)
    #         else:
    #             v += 1
    #     u += 1

    # step 1: divide into multiple text region candidates
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)

    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
        # if quadrilateral_can_merge_region_coarse(ubox, vbox):
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=2,
                                          char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    # step 2: postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
         region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # step 3: return regions
    for node_set in region_indices:
    # for node_set in nx.algorithms.components.connected_components(G):
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = np.array(bboxes)[nodes]

        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))

        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1 :
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1] : # if top 2 have the same counts
            max_aspect_ratio = -100
            for box in txtlns :
                if box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else :
            majority_dir = majority_dir_top_2[0][0]

        # sort textlines
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]

        # yield overall bbox and sorted indices
        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)

async def dispatch(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False) -> List[TextBlock]:
    # print(width, height)
    # import re
    # for l in textlines:
    #     s = str(l.pts)
    #     s = re.sub(r'([\d\]]) ', r'\1, ', s.replace('\n ', ', ')).replace(']]', ']],')
    #     print(s)

    text_regions: List[TextBlock] = []
    for (txtlns, fg_color, bg_color) in merge_bboxes_text_region(textlines, width, height):
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        total_logprobs /= sum([txtln.area for txtln in textlines])

        font_size = int(min([txtln.font_size for txtln in txtlns]))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        region = TextBlock(lines, texts, font_size=font_size, angle=angle, prob=np.exp(total_logprobs),
                           fg_color=fg_color, bg_color=bg_color)
        text_regions.append(region)
    return text_regions
