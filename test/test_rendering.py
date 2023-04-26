import os
import sys
import pytest
import numpy as np

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.rendering import remove_intersections

@pytest.mark.asyncio
async def test_intersection_removal1():
    points_list = [
        [[ 5,100],[50,100],[20,5],[ 5,5]],
        [[40,100],[80,100],[80,5],[40,5]],
    ]
    points_list = [np.array(pts) for pts in points_list]
    expected = [
        [],
    ]
    print(remove_intersections(points_list))

@pytest.mark.asyncio
async def test_intersection_removal2():
    points_list = [
        [[ 5,100],[50, 80],[55,30],[ 5,5]],
        [[40,100],[80,100],[80, 5],[40,5]],
    ]
    points_list = [np.array(pts) for pts in points_list]
    expected = [
        [],
    ]
    print(remove_intersections(points_list))
