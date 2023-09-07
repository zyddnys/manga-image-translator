import os
import sys
import cv2
import pytest
import numpy as np

pytest_plugins = ('pytest_asyncio')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.rendering import dispatch as dispatch_rendering, dispatch_eng_render
from manga_translator.utils import (
    TextBlock,
    visualize_textblocks,
)


RENDER_IMAGE_FOLDER = 'test/testdata/render'
os.makedirs(RENDER_IMAGE_FOLDER, exist_ok=True)

def save_result(path, img, regions):
    path = os.path.join(RENDER_IMAGE_FOLDER, path)
    cv2.imwrite(path, visualize_textblocks(img, regions))


@pytest.mark.asyncio
async def test_default_renderer():
    width, height = 1000, 1000
    img = np.zeros((height, width, 3))
    regions = [
        TextBlock(
            [[[10, 10], [200, 10], [10, 400], [200, 400]]],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[410, 10], [900, 10], [410, 800], [900, 800]]],
            translation=#'aaaaaa bbbbbbbbbbbb cccc' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd eeeeeeeeeeeeee fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd ' \
                'normal english sentences can be hyphenated! ' \
                'Pneumonoultramicroscopicsilicovolcanoconiosis'
        ),
    ]
    for region in regions:
        region.target_lang = 'ENG'
        region.set_font_colors([255, 255, 255], [200, 200, 200])
        region.font_size = 100

    img_rendered = await dispatch_rendering(img, regions, hyphenate=False)
    save_result('default1.png', img_rendered, regions)
