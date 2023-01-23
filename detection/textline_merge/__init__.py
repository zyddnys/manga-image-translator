import itertools
from collections import Counter
from typing import List, Set
import cv2
import numpy as np
import networkx as nx

from utils import Quadrilateral, quadrilateral_can_merge_region
from detection.ctd_utils import TextBlock

def split_text_region(bboxes: List[Quadrilateral], region_indices: Set[int], gamma = 0.5, sigma = 2, std_threshold = 6.0, verbose: bool = False) -> List[Set[int]]:
    region_indices = list(region_indices)
    if verbose:
        print('to split', region_indices)
    if len(region_indices) == 1:
        # case #1
        return [set(region_indices)]
    if len(region_indices) == 2:
        # case #2
        fs1 = bboxes[region_indices[0]].font_size
        fs2 = bboxes[region_indices[1]].font_size
        fs = max(fs1, fs2)
        if bboxes[region_indices[0]].distance(bboxes[region_indices[1]]) < (1 + gamma) * fs \
            and abs(bboxes[region_indices[0]].angle - bboxes[region_indices[1]].angle) < 4 * np.pi / 180:
            return [set(region_indices)]
        else:
            return [set([region_indices[0]]), set([region_indices[1]])]
    # case 3
    G = nx.Graph()
    for idx in region_indices:
        G.add_node(idx)
    for (u, v) in itertools.combinations(region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="kruskal", data=True)
    edges = sorted(edges, key=lambda a: a[2]['weight'], reverse=True)
    edge_weights = [a[2]['weight'] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in region_indices])
    std = np.std(edge_weights)
    mean = np.mean(edge_weights)
    if verbose:
        print('edge_weights', edge_weights)
        print(f'std: {std}, mean: {mean}')
    if (edge_weights[0] <= mean + std * sigma or edge_weights[0] <= fontsize * (1 + gamma)) and std < std_threshold:
        return [set(region_indices)]
    else:
        if edge_weights[0] - edge_weights[1] < std * sigma and std < std_threshold:
            return [set(region_indices)]
        if verbose:
            (split_u, split_v, _) = edges[0]
            print(f'split between "{bboxes[split_u].text}", "{bboxes[split_v].text}"')
        G = nx.Graph()
        for idx in region_indices:
            G.add_node(idx)
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(split_text_region(bboxes, node_set, verbose=verbose))
        return ans

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    box = np.array(box)
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - startidx, 0)
    box = np.array(box)
    return box

def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height, verbose):
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box = box)
        bboxes[i].assigned_index = i

    # step 1: divide into multiple text region candidates
    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=1.7,
                                          char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    # step 2: split each region
    #- region_indices: List[Set[int]] = []
    # for node_set in nx.algorithms.components.connected_components(G):
    #-     region_indices.extend(split_text_region(bboxes, node_set, verbose = verbose))
    # if verbose:
    #     print('region_indices', region_indices)

    for node_set in nx.algorithms.components.connected_components(G):
        nodes = list(node_set)
        txtlns = np.array(bboxes)[nodes]
        # get overall bbox
        # kq = np.concatenate([x.pts for x in txtlns], axis = 0)
        # if sum([int(a.is_approximate_axis_aligned) for a in txtlns]) > len(txtlns) // 2:
        #     max_coord = np.max(kq, axis = 0)
        #     min_coord = np.min(kq, axis = 0)
        #     merged_box = np.maximum(np.array([
        #         np.array([min_coord[0], min_coord[1]]),
        #         np.array([max_coord[0], min_coord[1]]),
        #         np.array([max_coord[0], max_coord[1]]),
        #         np.array([min_coord[0], max_coord[1]])
        #         ]), 0)
        #     bbox = np.concatenate([a[None, :] for a in merged_box], axis = 0).astype(int)
        # else:
        #     # TODO: use better method
        #     bbox = np.concatenate([a[None, :] for a in get_mini_boxes(kq)], axis = 0).astype(int)
        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))
        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir = Counter(dirs).most_common(1)[0][0]

        # sort
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))

        nodes = np.array(nodes) - min(nodes)
        for missing_value in reversed(list( set(range(max(nodes))).difference(set(nodes) ))):
            for i, node in enumerate(nodes):
                if node > missing_value:
                    nodes[i] = node - 1
        txtlns = [txtlns[i] for i in nodes]

        # yield overall bbox and sorted indices
        yield txtlns, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b

async def dispatch(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False) -> List[TextBlock]:
    text_regions: List[TextBlock] = []
    for (txtlns, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b) in merge_bboxes_text_region(textlines, width, height, verbose):
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        total_logprobs /= sum([txtln.area for txtln in textlines])

        x1 = min([txtln.aabb.x for txtln in txtlns])
        x2 = max([txtln.aabb.x + txtln.aabb.w for txtln in txtlns])
        y1 = min([txtln.aabb.y for txtln in txtlns])
        y2 = max([txtln.aabb.y + txtln.aabb.h for txtln in txtlns])
        font_size = min([txtln.font_size for txtln in txtlns])
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]

        region = TextBlock([x1, y1, x2, y2], lines, font_size=font_size, vertical=(majority_dir == 'v'), angle=angle,
                           fg_r=fg_r, fg_g=fg_g, fg_b=fg_b, bg_r=bg_r, bg_g=bg_g, bg_b=bg_b)
        region.prob = np.exp(total_logprobs)
        text_regions.append(region)
    return text_regions
