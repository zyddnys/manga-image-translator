import base64
import struct
from typing import Dict, List, Annotated

import cv2
import numpy as np
from pydantic import BaseModel, Field, WithJsonSchema

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
NumpyNdarray = Annotated[
    np.ndarray,
    WithJsonSchema({'type': 'string', "format": "base64","examples": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."]}),
]

class TextColor(BaseModel):
    fg: tuple[int, int, int]
    bg: tuple[int, int, int]

class Translation(BaseModel):
    minX: int
    minY: int
    maxX: int
    maxY: int
    is_bulleted_list: bool
    angle: float | int
    prob: float
    text_color: TextColor
    text: dict[str, str]
    background: NumpyNdarray = Field(
        ...,
        description="Background image encoded as a base64 string",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."]
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda array: Translation.encode_background(array)
        }

    @staticmethod
    def encode_background(array: np.ndarray) -> str:
        retval, buffer = cv2.imencode('.png', array)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        background = f"data:image/png;base64,{jpg_as_text}"
        return background

    def to_bytes(self):
        coords_bytes = struct.pack('4i', self.minX, self.minY, self.maxX, self.maxY)
        is_bulleted_list_byte = struct.pack('?', self.is_bulleted_list)
        angle_bytes = struct.pack('f', float(self.angle) if isinstance(self.angle, int) else self.angle)
        prob_bytes = struct.pack('f', self.prob)
        fg = struct.pack('3B', self.text_color.fg[0], self.text_color.fg[1], self.text_color.fg[2])
        bg = struct.pack('3B', self.text_color.bg[0], self.text_color.bg[1], self.text_color.bg[2])
        text_bytes = struct.pack('i', len(self.text.items()))
        for key, value in self.text.items():
            text_bytes += struct.pack('I', len(key.encode('utf-8'))) + key.encode('utf-8')
            text_bytes += struct.pack('I', len(value.encode('utf-8'))) + value.encode('utf-8')
        background_bytes = struct.pack('I', len(self.background.tobytes())) + self.background.tobytes()
        return coords_bytes +is_bulleted_list_byte+ angle_bytes+prob_bytes+fg + bg + text_bytes + background_bytes

class TranslationResponse(BaseModel):
    translations: List[Translation]

    def to_bytes(self):
        items= [v.to_bytes() for v in self.translations]
        return struct.pack('i', len(items)) + b''.join(items)

def to_translation(ctx: Context) -> TranslationResponse:
    text_regions:list[TextBlock] = ctx.text_regions
    inpaint = ctx.img_inpainted
    translations:Dict[str, List[str]] = ctx.translations
    results = []
    for i, blk in enumerate(text_regions):
        minX, minY, maxX, maxY = blk.xyxy
        text_region = text_regions[i]
        if 'translations' in ctx:
            trans = {key: value[i] for key, value in translations.items()}
        else:
            trans = {}
        trans[text_region.source_lang] = text_regions[i].text
        text_region.adjust_bg_color = False
        color1, color2 = text_region.get_font_colors()
        results.append(Translation(text=trans,
                    minX=int(minX),minY=int(minY),maxX=int(maxX),maxY=int(maxY),
                    background=inpaint[minY:maxY, minX:maxX],
                    is_bulleted_list=text_region.is_bulleted_list,
                    text_color=TextColor(fg=color1.tolist(), bg=color2.tolist()),
                    prob=text_region.prob,
                    angle=text_region.angle
        ))
        #todo: background angle
    return TranslationResponse(translations=results)
