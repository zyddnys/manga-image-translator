import os
import sys
import cv2
import pytest
import numpy as np

from manga_translator.rendering import dispatch as dispatch_rendering, dispatch_eng_render, text_render
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
            texts=['a', 'b','c', 'd', 'e', 'f'],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[410, 10], [900, 10], [410, 800], [900, 800]]],
            texts=['eng', 'pne'],
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


ARABIC = 'مرحبا يا صديقي، هل أنت مستعد للمغامرة؟'

def test_arabic_wraps_in_reading_order():
    text_render.set_font('')
    lines, widths = text_render.calc_horizontal_rtl(48, ARABIC, 300, 140)
    # Lines break at spaces only, keep the space after the comma,
    # and the sentence starts on the first line
    assert len(lines) > 1
    assert ' '.join(lines) == ARABIC
    assert lines[0].startswith('مرحبا')
    assert widths == [text_render.get_rtl_string_width(48, line) for line in lines]

def test_arabic_translation_is_rendered_as_is():
    # Shaping and right-to-left layout happen in the renderer, so the
    # translation keeps its logical order, Latin words and digits included
    region = TextBlock([[[10, 10], [200, 10], [10, 400], [200, 400]]], texts=['a'], translation='اقرأ manga كل يوم 3 مرات')
    region.target_lang = 'ARA'
    assert region.get_translation_for_rendering() == 'اقرأ manga كل يوم 3 مرات'

@pytest.mark.asyncio
async def test_default_renderer_arabic():
    width, height = 600, 600
    img = np.zeros((height, width, 3))
    region = TextBlock([[[10, 10], [500, 10], [10, 300], [500, 300]]], texts=['a'], translation=ARABIC)
    region.target_lang = 'ARA'
    region.set_font_colors([255, 255, 255], [0, 0, 0])
    region.font_size = 48

    blank = img.copy()
    img_rendered = await dispatch_rendering(img, [region], hyphenate=False)
    assert (img_rendered != blank).any()
    save_result('default_arabic.png', img_rendered, [region])
