import numpy as np
from abc import abstractmethod
from typing import List, Union
from collections import Counter
import networkx as nx
import itertools
import os
import cv2

from ..utils import InfererModule, TextBlock, ModelWrapper, Quadrilateral

class CommonOCR(InfererModule):
    def _generate_text_direction(self, bboxes: List[Union[Quadrilateral, TextBlock]]):
        if len(bboxes) > 0:
            if isinstance(bboxes[0], TextBlock):
                for blk in bboxes:
                    for line_idx in range(len(blk.lines)):
                        yield blk, line_idx
            else:
                from ..utils import quadrilateral_can_merge_region

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

    async def recognize(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        '''
        Performs the optical character recognition, using the `textlines` as areas of interests.
        Returns a `textlines` list with the `textline.text` property set to the detected text string.
        '''
        return await self._recognize(image, textlines, verbose)

    @abstractmethod
    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        pass

    def check_color(self,image):
        """
        Determine whether there are colors in non black, gray, white, and other gray areas in an RGB color image。
        params：
        image -- np.array
        return：
        True -- Colors with non black, gray, white, and other grayscale areas
        False -- Images are all grayscale areas
        """
        gray_image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                color = image[i, j]
                color_distance = np.sum((color - gray_image[i, j])**2)
                if color_distance > 100:
                    return True
        return False

    def is_ignore(self,region_img):
        """
        Principle: Normally, white bubbles and their text boxes are mostly white, while black bubbles and their text boxes are mostly black. We calculate the ratio of white or black pixels around the text block to the total pixels, and judge whether the area is a normal bubble area or not. Based on the value of the --ingore-bubble parameter, if the ratio is greater than the base value and less than (100-base value), then it is considered a non-bubble area.
        The normal range for ingore-bubble is 1-50, and other values are considered not input. The recommended value for ingore-bubble is 10. The smaller it is, the more likely it is to recognize normal bubbles as image text and skip them. The larger it is, the more likely it is to recognize image text as normal bubbles.

        Assuming ingore-bubble = 10
        The text block is surrounded by white if it is <10, and the text block is very likely to be a normal white bubble.
        The text block is surrounded by black if it is >90, and the text block is very likely to be a normal black bubble.
        Between 10 and 90, if there are black and white spots around it, the text block is very likely not a normal bubble, but an image.

        The input parameter is the image data of the text block processed by OCR.
        Calculate the ratio of black or white pixels in the four rectangular areas formed by taking 2 pixels from the edges of the four sides of the image.
        Return the overall ratio. If it is between basevalue and (100-basevalue), skip it.

        last determine if there is color, consider the colored text as invalid information and skip it without translation
        """
        basevalue=int(os.environ['ignore_bubble'])
        self.logger.info(f"\nignore_bubble:{basevalue}")

        if basevalue<1 or basevalue>50:
            self.logger.info(f"ignore_bubble not between 1 and 99, no need to handle")
            return  False
        # 255 is white, 0 is black
        _, binary_raw_mask = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)
        height, width = binary_raw_mask.shape[:2]

        total=0
        top_sum = sum(binary_raw_mask[0:2, 0:width].ravel() == 0)
        total+= binary_raw_mask[0:2, 0:width].size

        bottom_sum = sum(binary_raw_mask[height-2:height, 0:width].ravel() == 0)
        total+= binary_raw_mask[height-2:height, 0:width].size

        left_sum = sum(binary_raw_mask[2:height-2, 0:2].ravel() == 0)
        total+= binary_raw_mask[2:height-2, 0:2].size

        right_sum = sum(binary_raw_mask[2:height-2, width-2:width].ravel() == 0)
        total += binary_raw_mask[2:height-2, width-2:width].size

        sum_all=top_sum+bottom_sum+left_sum+right_sum
        ratio = round( sum_all / total, 6)*100
        self.logger.info(f"ingore:sum_all={top_sum},total={total},ratio={ratio}")
        if ratio>=basevalue and ratio<=(100-basevalue):
            self.logger.info(f"ignore this text block")
            return True
        # To determine if there is color, consider the colored text as invalid information and skip it without translation
        if self.check_color(region_img):
            self.logger.info(f"ignore Colorful text block")
            return True
        self.logger.info(f"normal bubble")
        return False


class OfflineOCR(CommonOCR, ModelWrapper):
    _MODEL_SUB_DIR = 'ocr'

    async def _recognize(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], verbose: bool = False) -> List[Quadrilateral]:
        pass
