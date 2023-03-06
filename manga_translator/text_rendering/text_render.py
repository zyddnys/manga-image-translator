import os
import cv2
import numpy as np
import unicodedata
import freetype
import functools
from typing import Tuple, Optional, List

from ..utils import BASE_PATH

def _is_whitespace(ch):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False

def _is_control(ch):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _is_punctuation(ch):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(ch)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False

def count_valuable_text(text) -> int:
    return sum([1 for ch in text if not _is_punctuation(ch) and not _is_control(ch) and not _is_whitespace(ch)])


CJK_H2V = {
    "‥" :"︰" ,
    "—" :"︱" ,
    "–" :"︲" ,
    "_" :"︳" ,
    "_" :"︴",
    "(" :"︵" ,
    ")" :"︶" ,
    "（" :"︵" ,
    "）" :"︶" ,
    "{" :"︷" ,
    "}" :"︸" ,
    "〔":"︹" ,
    "〕":"︺" ,
    "【":"︻" ,
    "】":"︼" ,
    "《":"︽" ,
    "》":"︾" ,
    "〈":"︿" ,
    "〉":"﹀" ,
    "「":"﹁" ,
    "」":"﹂" ,
    "『":"﹃" ,
    "』":"﹄" ,
    "﹑":"﹅",
    "﹆" :"﹆" ,
    "[" :"﹇" ,
    "]" :"﹈" ,
    "﹉":"﹉",
    "﹊":"﹊",
    "﹋":"﹋",
    "﹌":"﹌",
    "﹍":"﹍",
    "﹎":"﹎",
    "﹏":"﹏",
    "…": "⋮"
}

CJK_V2H = {
    "︰" :"‥" ,
    "︱" :"—" ,
    "︲" :"–" ,
    "︳" :"_" ,
    "︴":"_" ,
    "︵" :"(" ,
    "︶" :")" ,
    "︷" :"{" ,
    "︸" :"}" ,
    "︹" :"〔",
    "︺" :"〕",
    "︻" :"【",
    "︼" :"】",
    "︽" :"《",
    "︾" :"》",
    "︿" :"〈",
    "﹀" :"〉",
    "﹁" :"「",
    "﹂" :"」",
    "﹃" :"『",
    "﹄" :"』",
    "﹅":"﹑",
    "﹆" :"﹆" ,
    "﹇" :"[" ,
    "﹈" :"]" ,
    "﹉":"﹉",
    "﹊":"﹊",
    "﹋":"﹋",
    "﹌":"﹌",
    "﹍":"﹍",
    "﹎":"﹎",
    "﹏":"﹏",
    "⋮": "…"
}

def CJK_Compatibility_Forms_translate(cdpt: str, direction: int):
    if cdpt == 'ー' and direction == 1:
        return 'ー', 90
    if cdpt in ["︰", "︱", "︲", "︳", "︴", "︵", "︶", "︷", "︸", "︹", "︺", "︻", "︼", "︽", "︾", "︿", "﹀", "﹁", "﹂", "﹃", "﹄", "﹅", "﹆", "﹇", "﹈", "﹉", "﹊", "﹋", "﹌", "﹍", "﹎", "﹏", "⋮"]:
        if direction == 0:
            # translate
            return CJK_V2H[cdpt], 0
        else:
            return cdpt, 0
    elif cdpt in ["‥", "—", "–", "_", "_", "(", ")", "（", "）", "{", "}", "〔", "〕", "【", "】", "《", "》", "〈", "〉", "「", "」", "『", "』", "﹑", "﹆", "[", "]", "﹉", "﹊", "﹋", "﹌", "﹍", "﹎", "﹏", "…"]:
        if direction == 1:
            # translate
            return CJK_H2V[cdpt], 0
        else:
            return cdpt, 0
    return cdpt, 0

def compact_special_symbols(text: str) -> str :
    text = text.replace('...', '…')
    return text

def rotate_image(image, angle):
    if angle == 0:
        return image, (0, 0)
    image_exp = np.zeros((round(image.shape[0] * 1.5), round(image.shape[1] * 1.5), image.shape[2]), dtype = np.uint8)
    diff_i = (image_exp.shape[0] - image.shape[0]) // 2
    diff_j = (image_exp.shape[1] - image.shape[1]) // 2
    image_exp[diff_i:diff_i+image.shape[0], diff_j:diff_j+image.shape[1]] = image
    # from https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    image_center = tuple(np.array(image_exp.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image_exp, rot_mat, image_exp.shape[1::-1], flags=cv2.INTER_LINEAR)
    if angle == 90:
        return result, (0, 0)
    return result, (diff_i, diff_j)

def add_color(bw_char_map, color, stroke_char_map, stroke_color):
    fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype = np.uint8)
    if bw_char_map.size == 0:
        fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 3), dtype = np.uint8)
        return fg.astype(np.uint8)
    fg[:,:,0] = color[0]
    fg[:,:,1] = color[1]
    fg[:,:,2] = color[2]
    fg[:,:,3] = bw_char_map

    bg = np.zeros((stroke_char_map.shape[0], stroke_char_map.shape[1], 4), dtype = np.uint8)
    bg[:,:,0] = stroke_color[0]
    bg[:,:,1] = stroke_color[1]
    bg[:,:,2] = stroke_color[2]
    bg[:,:,3] = stroke_char_map

    fg_alpha = fg[:, :, 3] / 255.0
    bg_alpha = 1.0 - fg_alpha
    bg[:, :, :] = (fg_alpha[:, :, np.newaxis] * fg[:, :, :] + bg_alpha[:, :, np.newaxis] * bg[:, :, :])
    #alpha_char_map = cv2.add(bw_char_map, stroke_char_map)
    #alpha_char_map[alpha_char_map > 0] = 255
    return bg#, alpha_char_map

FALLBACK_FONTS = [
    os.path.join(BASE_PATH, 'fonts/Arial-Unicode-Regular.ttf'),
    os.path.join(BASE_PATH, 'fonts/msyh.ttc'),
    os.path.join(BASE_PATH, 'fonts/msgothic.ttc'),
]
FONT_SELECTION: List[freetype.Face] = []
font_cache = {}
def get_cached_font(path: str) -> freetype.Face:
    if not font_cache.get(path):
        font_cache[path] = freetype.Face(path)
    return font_cache[path]

def set_font(font_path: str):
    global FONT_SELECTION
    if font_path:
        selection = [font_path] + FALLBACK_FONTS
    else:
        selection = FALLBACK_FONTS
    FONT_SELECTION = [get_cached_font(p) for p in selection]

class namespace:
    pass

class Glyph:
    def __init__(self, glyph):
        self.bitmap = namespace()
        self.bitmap.buffer = glyph.bitmap.buffer
        self.bitmap.rows = glyph.bitmap.rows
        self.bitmap.width = glyph.bitmap.width
        self.advance = namespace()
        self.advance.x = glyph.advance.x
        self.advance.y = glyph.advance.y
        self.bitmap_left = glyph.bitmap_left
        self.bitmap_top = glyph.bitmap_top
        self.metrics = namespace()
        self.metrics.vertBearingX = glyph.metrics.vertBearingX
        self.metrics.vertBearingY = glyph.metrics.vertBearingY
        self.metrics.horiBearingX = glyph.metrics.horiBearingX
        self.metrics.horiBearingY = glyph.metrics.horiBearingY
        self.metrics.horiAdvance = glyph.metrics.horiAdvance
        self.metrics.vertAdvance = glyph.metrics.vertAdvance

@functools.lru_cache(maxsize = 1024, typed = True)
def get_char_glyph(cdpt: str, font_size: int, direction: int) -> Glyph:
    global FONT_SELECTION
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        elif direction == 1:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt)
        return Glyph(face.glyph)

#@functools.lru_cache(maxsize = 1024, typed = True)
def get_char_border(cdpt: str, font_size: int, direction: int):
    global FONT_SELECTION
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        elif direction == 1:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
        slot_border = face.glyph
        return slot_border.get_glyph()

# def get_char_kerning(cdpt, prev, font_size: int, direction: int):
#     global FONT_SELECTION
#     for i, face in enumerate(FONT_SELECTION):
#         if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
#             continue
#         if direction == 0:
#             face.set_pixel_sizes(0, font_size)
#         elif direction == 1:
#             face.set_pixel_sizes(font_size, 0)
#         face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
#         #print("VV", prev, cdpt, face.get_char_index(prev), face.get_char_index(cdpt))
#         print("VR", face.has_kerning)
#         return face.get_kerning(face.get_char_index(prev), face.get_char_index(cdpt))

def calc_vertical(font_size: int, text: str, max_height: int, spacing_x: int):
    line_text_list = []
    line_width_list = []
    line_height_list = []

    line_str = ""
    line_height = 0
    line_width_left = 0
    line_width_right = 0
    for i, cdpt in enumerate(text):
        is_pun = _is_punctuation(cdpt)
        cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
        slot = get_char_glyph(cdpt, font_size, 1)
        bitmap = slot.bitmap
        # spaces, etc
        if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
            char_offset_y = slot.metrics.vertBearingY >> 6
        else:
            char_offset_y = slot.metrics.vertAdvance >> 6
        char_width = bitmap.width
        char_bearing_x = slot.metrics.vertBearingX >> 6
        if line_height + char_offset_y > max_height:
            line_text_list.append(line_str)
            line_height_list.append(line_height)
            line_width_list.append(line_width_left + line_width_right)
            line_str = ""
            line_height = 0
            line_width_left = 0
            line_width_right = 0
        line_height += char_offset_y
        line_str += cdpt
        line_width_left = max(line_width_left, abs(char_bearing_x))
        line_width_right = max(line_width_right, char_width - abs(char_bearing_x))
    # last char
    line_text_list.append(line_str)
    line_height_list.append(line_height)
    line_width_list.append(line_width_left + line_width_right)

    box_calc_x = sum(line_width_list) + (len(line_width_list) - 1) * spacing_x
    box_calc_y = max(line_height_list)
    return line_text_list, box_calc_x, box_calc_y

def put_char_vertical(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int):
    pen = pen_l.copy()

    is_pun = _is_punctuation(cdpt)
    cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
    slot = get_char_glyph(cdpt, font_size, 1)
    bitmap = slot.bitmap
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        char_offset_y = slot.metrics.vertBearingY >> 6
        return char_offset_y
    char_offset_y = slot.metrics.vertAdvance >> 6
    bitmap_char = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
    pen[0] += slot.metrics.vertBearingX >> 6
    pen[1] += slot.metrics.vertBearingY >> 6
    canvas_text[pen[1]:pen[1]+bitmap.rows, pen[0]:pen[0]+bitmap.width] = bitmap_char
    #print(pen_l, pen, slot.metrics.vertBearingX >> 6, bitmap.width)
    #border
    if border_size > 0:
        pen_border = (max(pen[0] - border_size, 0), max(pen[1] - border_size, 0))
        #slot_border = 
        glyph_border = get_char_border(cdpt, font_size, 1)
        stroker = freetype.Stroker()
        stroker.set(64 * max(int(0.07 * font_size), 1), freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINEJOIN_ROUND, 0)
        glyph_border.stroke(stroker, destroy=True)
        blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0,0), True)
        bitmap_b = blyph.bitmap
        bitmap_border = np.array(bitmap_b.buffer, dtype = np.uint8).reshape(bitmap_b.rows,bitmap_b.width)
        canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width] = cv2.add(canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width], bitmap_border)
    return char_offset_y

def put_text_vertical(font_size: int, text: str, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]):
    text = compact_special_symbols(text)
    bgsize = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = 0
    spacing_x = int(font_size * 0.2)

    # make large canvas
    num_char_y = h // font_size
    num_char_x = len(text) // num_char_y + 1
    canvas_x = font_size * num_char_x + spacing_x * (num_char_x - 1) + (font_size + bgsize) * 2
    canvas_y = font_size * num_char_y + (font_size + bgsize) * 2
    ##line_text_list, canvas_x, canvas_y = calc_vertical(font_size, text, h, spacing_x=spacing_x)
    canvas_text = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [canvas_text.shape[1] - (font_size + bgsize), font_size + bgsize]
    line_height = 0
    # write stuff
    for t in text:
        if line_height == 0:
            pen_line = pen_orig.copy()
            if t == ' ':
                continue
        offset_y = put_char_vertical(font_size, t, pen_line, canvas_text, canvas_border, border_size=bgsize)
        line_height += offset_y
        if line_height + font_size > h:
            pen_orig[0] -= spacing_x + font_size
            line_height = 0
        else:
            pen_line[1] += offset_y

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, w, h = cv2.boundingRect(canvas_border)
    return line_box[y:y+h, x:x+w]

def calc_horizontal(font_size: int, text: str, limit_width: int) -> Tuple[List[str], List[int]]:
    line_text_list = []
    line_width_list = []
    line_str = ""
    line_width = 0
    word_str = ""
    word_width = 0
    max_width = limit_width + font_size
    space = False

    # 1. JPN, CHN : left-align, no spaces, confine to limit_width
    previous_cdpt = ''
    for i, cdpt in enumerate(text):
        is_pun = _is_punctuation(cdpt)
        cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 0)
        glyph = get_char_glyph(cdpt, font_size, 0)
        bitmap = glyph.bitmap
        # next_glyph = get_char_glyph(text[min(i+1, len(text)-1)], font_size, 0)
        # next_bitmap = next_glyph.bitmap
        next_is_space = _is_whitespace(text[min(i+1, len(text)-1)]) or _is_punctuation(text[min(i+1, len(text)-1)])
        # spaces, etc
        if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
            char_offset_x = glyph.advance.x >> 6
            space = True
        else:
            char_offset_x = glyph.metrics.horiAdvance >> 6
            space = False

        if space:
            if line_width + word_width > limit_width or word_width > limit_width:
                if len(line_str.strip()) > 0: # make sure not to add empty lines
                    line_text_list.append(line_str.strip())
                    line_width_list.append(line_width)
                line_str = ""
                line_width = 0
            line_str += word_str
            line_width += word_width
            word_width = 0
            word_str = ""
        if line_width + word_width + char_offset_x > limit_width: # force line break mid word
            if len(word_str) <= 6 or next_is_space: # word is too short or next char would be a space anyway
                # clear the current line and start a new one
                if len(line_str.strip()) > 0: # make sure not to add empty lines
                    line_text_list.append(line_str.strip())
                    line_width_list.append(line_width)
                line_str = ""
                line_width = 0
            else:
                # add "-" to a word break
                word_str += "-"
                word_width += char_offset_x
                line_str += word_str
                line_width += word_width
                word_width = 0
                word_str = ""
                line_text_list.append(line_str.strip())
                line_width_list.append(line_width)
                line_str = ""
                line_width = 0
        word_str += cdpt
        word_width += char_offset_x
        previous_cdpt = cdpt

    # last char
    line_str += word_str
    line_width += word_width
    line_text_list.append(line_str.strip())
    line_width_list.append(line_width)

    # 2. ELSE : center-align, break on spaces, can reach max_width if necessary (one word)

    return line_text_list, line_width_list

def put_char_horizontal(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int):
    pen = pen_l.copy()

    # is_pun = _is_punctuation(cdpt)
    cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 0)
    slot = get_char_glyph(cdpt, font_size, 0)
    bitmap = slot.bitmap
    char_offset_x = slot.advance.x >> 6
    bitmap_char = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return char_offset_x
    pen[0] += slot.bitmap_left
    pen[1] = max(pen[1] - slot.bitmap_top, 0)
    canvas_text[pen[1]:pen[1]+bitmap.rows, pen[0]:pen[0]+bitmap.width] = bitmap_char
    #print(pen_l, pen, slot.metrics.vertBearingX >> 6, bitmap.width)
    #border
    if border_size > 0:
        pen_border = (max(pen[0] - border_size, 0), max(pen[1] - border_size, 0))
        #slot_border = 
        glyph_border = get_char_border(cdpt, font_size, 1)
        stroker = freetype.Stroker()
        stroker.set(64 * max(int(0.07 * font_size), 1), freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINEJOIN_ROUND, 0)
        glyph_border.stroke(stroker, destroy=True)
        blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0,0), True)
        bitmap_b = blyph.bitmap
        bitmap_border = np.array(bitmap_b.buffer, dtype = np.uint8).reshape(bitmap_b.rows,bitmap_b.width)

        canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width] = cv2.add(canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width], bitmap_border)
    return char_offset_x

def put_text_horizontal(font_size: int, text: str, width: int, alignment: str, fg: Tuple[int, int, int], bg: Tuple[int, int, int]):
    text = compact_special_symbols(text)
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = int(font_size * 0.2)

    # calc
    line_text_list, line_width_list = calc_horizontal(font_size, text, width)
    # print(line_text_list, line_width_list)

    # make large canvas
    canvas_w = max(line_width_list) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [font_size + bg_size, font_size + bg_size]

    # write stuff
    for line_text, line_width in zip(line_text_list, line_width_list):
        pen_line = pen_orig.copy()
        if alignment == 'center':
            pen_line[0] += (max(line_width_list) - line_width) // 2
        elif alignment == 'right':
            pen_line[0] += max(line_width_list) - line_width
        for c in line_text:
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, width, height = cv2.boundingRect(canvas_border)
    return line_box[y:y+height, x:x+width]

# def put_text(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]):
#     pass

def test():
    #canvas = put_text_vertical(64, 1.0, '因为不同‼ [这"真的是普]通的》肉！那个“姑娘”的恶作剧！是吗？咲夜⁉。', 700, (0, 0, 0), (255, 128, 128))
    canvas = put_text_horizontal(64, 1.0, '因为不同‼ [这"真的是普]通的》肉！那个“姑娘”的恶作剧！是吗？咲夜⁉', 400, (0, 0, 0), (255, 128, 128))
    cv2.imwrite('text_render_combined.png', canvas)

if __name__ == '__main__':
    test()
