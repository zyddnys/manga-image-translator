import base64
from typing import Dict, List

import cv2
from pydantic import BaseModel

from manga_translator import Context
from manga_translator.utils import TextBlock


#input:PIL,
#result:PIL
#img_colorized: PIL
#upscaled:PIL
#img_rgb:array
#img_alpha:None
#textlines:list[Quadrilateral]
#text_regions:list[TextBlock]
#translations: map[str, arr[str]]
#img_inpainted: array
#gimp_mask:array
#img_rendered: array
#mask_raw: array
#mask:array
def to_json(ctx: Context):
    text_regions:list[TextBlock] = ctx.text_regions
    inpaint = ctx.img_inpainted
    translations:Dict[str, List[str]] = ctx.translations
    results = []
    if 'overlay_ext' in ctx:
        #todo: unreachable
        overlay_ext = ctx['overlay_ext']
    else:
        overlay_ext = 'jpg'
    for i, blk in enumerate(text_regions):
        minX, minY, maxX, maxY = blk.xyxy
        if 'translations' in ctx:
            trans = {key: value[i] for key, value in translations.items()}
        else:
            trans = {}
        trans["originalText"] = text_regions[i].text
        if inpaint is not None:
            overlay = inpaint[minY:maxY, minX:maxX]

            retval, buffer = cv2.imencode('.' + overlay_ext, overlay)
            jpg_as_text = base64.b64encode(buffer)
            background = "data:image/" + overlay_ext + ";base64," + jpg_as_text.decode("utf-8")
        else:
            background = None
        text_region = text_regions[i]
        text_region.adjust_bg_color = False
        color1, color2 = text_region.get_font_colors()

        results.append({
            'text': trans,
            'minX': int(minX),
            'minY': int(minY),
            'maxX': int(maxX),
            'maxY': int(maxY),
            'textColor': {
                'fg': color1.tolist(),
                'bg': color2.tolist()
            },
            'language': text_regions[i].source_lang,
            'background': background
        })
    return results

class TextColor(BaseModel):
    fg: tuple[int, int, int]
    bg: tuple[int, int, int]

class Translation(BaseModel):
    text: dict[str, str]
    minX: int
    minY: int
    maxX: int
    maxY: int
    textColor:TextColor
    language: str
    background: str
