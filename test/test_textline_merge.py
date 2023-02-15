import os
import sys
import cv2
import pytest

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.detection.textline_merge import *
from manga_translator.utils import (
    TextBlock,
    Quadrilateral,
    visualize_textblocks,
)


BBOX_IMAGE_FOLDER = 'test/testdata/bboxes'
os.makedirs(BBOX_IMAGE_FOLDER, exist_ok=True)

def save_bboxes(path: str, regions: TextBlock, width: int, height: int):
    img = np.zeros((height, width, 3))
    cv2.imwrite(path, visualize_textblocks(img, regions))

def calc_region_line_combinations(regions, lines):
    """
    Creates sorted combinations of indices from the lines depending on how they are
    ordered within the regions.
    """
    for i, region in enumerate(regions):
        combination = []
        for line1 in region.lines:
            for j, line2 in enumerate(lines):
                if (line1 == line2).all():
                    combination.append(j)
                    break
            else:
                raise Exception('Line was discarded')
        combination.sort()
        yield i, combination

async def run_test(lines, expected_combinations, width, height):
    quadrilaterals = [Quadrilateral(line, '', 1) for line in lines]
    regions = await dispatch(quadrilaterals, width, height)

    for i, line_combination in calc_region_line_combinations(regions, lines):
        for expected in expected_combinations:
            if line_combination == expected: # list comparision
                break
        else:
            p = os.path.join(BBOX_IMAGE_FOLDER, 'bboxes.png')
            save_bboxes(p, regions, width, height)
            raise Exception(f'Invalid bbox: "{i}" - Image saved under {p}')


@pytest.mark.asyncio
async def test_demo3():
    width, height = 2590, 4096
    lines = [
        np.array([[   0, 3280], [ 237, 3234], [ 394, 4069], [ 149, 4096]]),
        np.array([[2400, 3210], [2493, 3210], [2498, 4061], [2405, 4061]]),
        np.array([[2306, 3208], [2410, 3208], [2416, 3992], [2312, 3992]]),
        np.array([[2226, 3208], [2328, 3208], [2328, 4050], [2226, 4050]]),
        np.array([[2149, 3205], [2242, 3205], [2237, 4005], [2144, 4005]]),
        np.array([[2160, 2298], [2245, 2298], [2250, 3069], [2165, 3069]]),
        np.array([[2082, 2296], [2176, 2296], [2176, 3032], [2082, 3032]]),
        np.array([[2008, 2293], [2109, 2293], [2109, 2680], [2008, 2680]]),
        np.array([[ 162, 1733], [ 256, 1733], [ 256, 2141], [ 162, 2141]]),
        np.array([[ 242, 1733], [ 336, 1733], [ 336, 2144], [ 242, 2144]]),
        np.array([[2269, 1349], [2368, 1349], [2373, 1960], [2274, 1960]]),
        np.array([[2186, 1352], [2288, 1352], [2288, 1760], [2186, 1760]]),
        np.array([[2373, 1357], [2442, 1357], [2442, 2077], [2373, 2077]]),
        np.array([[ 536, 1349], [ 613, 1349], [ 613, 1997], [ 536, 1997]]),
        np.array([[ 594, 1344], [ 680, 1344], [ 696, 2072], [ 610, 2072]]),
        np.array([[1037,  485], [1282,  469], [1349, 1418], [1104, 1434]]),
        np.array([[ 234,  528], [ 312,  528], [ 312, 1176], [ 234, 1176]]),
        np.array([[ 138,  509], [ 256,  509], [ 256,  706], [ 138,  706]]),
        np.array([[2418,  384], [2504,  384], [2509, 1234], [2424, 1234]]),
        np.array([[2344,  381], [2429,  381], [2434,  965], [2349,  965]]),
        np.array([[2269,  376], [2370,  376], [2370,  818], [2269,  818]]),
        np.array([[ 197,   42], [2405,   37], [2405,  362], [ 197,  368]]),
    ]
    expected_combinations = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7],
        [8, 9],
        [10, 11, 12],
        [13, 14],
        [15],
        [16, 17],
        [18, 19, 20],
        [21],
    ]
    await run_test(lines, expected_combinations, width, height)
