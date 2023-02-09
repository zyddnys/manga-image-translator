import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter
import networkx as nx
import itertools

from ..utils import ModelWrapper, Quadrilateral
from ..detection.ctd_utils import TextBlock

class CommonOCR(ABC):
    def generate_text_direction(self, bboxes: List[Union[Quadrilateral, TextBlock]]):
        if len(bboxes) > 0:
            if isinstance(bboxes[0], TextBlock):
                for blk in bboxes:
                    for line_idx in range(len(blk.lines)):
                        yield blk, line_idx
            else:
                from utils import quadrilateral_can_merge_region

                G = nx.Graph()
                for i, box in enumerate(bboxes):
                    G.add_node(i, box = box)
                for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
                    if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1):
                        G.add_edge(u, v)
                for node_set in nx.algorithms.components.connected_components(G):
                    nodes = list(node_set)
                    # majority vote for direction
                    dirs = [box.direction for box in [bboxes[i] for i in nodes]]
                    majority_dir = Counter(dirs).most_common(1)[0][0]
                    # sort
                    if majority_dir == 'h':
                        nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
                    elif majority_dir == 'v':
                        nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
                    # yield overall bbox and sorted indices
                    for node in nodes:
                        yield bboxes[node], majority_dir

    async def recognize(self, image: np.ndarray, regions: List[TextBlock], verbose: bool = False) -> List[TextBlock]:
        '''
        Performs the optical character recognition, using the `regions` as areas of interests.
        Returns TextBlocks defined by the `regions` which contain the recognized text.
        '''
        return await self._recognize(image, regions, verbose)

    @abstractmethod
    async def _recognize(self, image: np.ndarray, regions: List[TextBlock], verbose: bool = False) -> List[TextBlock]:
        pass

class OfflineOCR(CommonOCR, ModelWrapper):
    _MODEL_DIR = os.path.join(ModelWrapper._MODEL_DIR, 'ocr')

    async def _recognize(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    @abstractmethod
    async def _forward(self, image: np.ndarray, regions: List[TextBlock], verbose: bool = False) -> List[TextBlock]:
        pass
