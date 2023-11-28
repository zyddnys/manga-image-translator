import os
import re
import cv2
import numpy as np
import freetype
import functools
from pathlib import Path
from typing import Tuple, Optional, List
from hyphen import Hyphenator
from hyphen.dictools import LANGUAGES as HYPHENATOR_LANGUAGES
from langcodes import standardize_tag

from ..utils import BASE_PATH, is_punctuation, is_whitespace

try:
    HYPHENATOR_LANGUAGES.remove('fr')
    HYPHENATOR_LANGUAGES.append('fr_FR')
except Exception:
    pass

CJK_H2V = {
    "‥": "︰",
    "—": "︱",
    "―": "|",
    "–": "︲",
    "_": "︳",
    "_": "︴",
    "(": "︵",
    ")": "︶",
    "（": "︵",
    "）": "︶",
    "{": "︷",
    "}": "︸",
    "〔": "︹",
    "〕": "︺",
    "【": "︻",
    "】": "︼",
    "《": "︽",
    "》": "︾",
    "〈": "︿",
    "〉": "﹀",
    "「": "﹁",
    "」": "﹂",
    "『": "﹃",
    "』": "﹄",
    "﹑": "﹅",
    "﹆": "﹆",
    "[": "﹇",
    "]": "﹈",
    "﹉": "﹉",
    "﹊": "﹊",
    "﹋": "﹋",
    "﹌": "﹌",
    "﹍": "﹍",
    "﹎": "﹎",
    "﹏": "﹏",
    "…": "⋮",
}

CJK_V2H = {
    **dict(zip(CJK_H2V.items(), CJK_H2V.keys())),
}

def CJK_Compatibility_Forms_translate(cdpt: str, direction: int):
    """direction: 0 - horizontal, 1 - vertical"""
    if cdpt == 'ー' and direction == 1:
        return 'ー', 90
    if cdpt in CJK_V2H:
        if direction == 0:
            # translate
            return CJK_V2H[cdpt], 0
        else:
            return cdpt, 0
    elif cdpt in CJK_H2V:
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
    if bw_char_map.size == 0:
        fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype = np.uint8)
        return fg
    
    # print(bw_char_map.shape, stroke_char_map.shape)
    # import matplotlib.pyplot as plt
    # x1, y1, w1, h1 = cv2.boundingRect(bw_char_map)
    # x2, y2, w2, h2 = cv2.boundingRect(stroke_char_map)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(bw_char_map)
    # ax[1].imshow(stroke_char_map)
    # # draw bounding boxes
    # rect1 = plt.Rectangle((x1, y1), w1, h1, fill=False, color='red')
    # rect2 = plt.Rectangle((x2, y2), w2, h2, fill=False, color='blue')
    # ax[0].add_patch(rect1)
    # ax[0].add_patch(rect2)
    # rect1 = plt.Rectangle((x1, y1), w1, h1, fill=False, color='red')
    # rect2 = plt.Rectangle((x2, y2), w2, h2, fill=False, color='blue')
    # ax[1].add_patch(rect1)
    # ax[1].add_patch(rect2)
    # plt.show()

    # since bg rect is always larger than fg rect, we can just use the bg rect
    x, y, w, h = cv2.boundingRect(stroke_char_map)

    fg = np.zeros((h, w, 4), dtype = np.uint8)
    fg[:,:,0] = color[0]
    fg[:,:,1] = color[1]
    fg[:,:,2] = color[2]
    fg[:,:,3] = bw_char_map[y:y+h, x:x+w]

    bg = np.zeros((stroke_char_map.shape[0], stroke_char_map.shape[1], 4), dtype = np.uint8)
    bg[:,:,0] = stroke_color[0]
    bg[:,:,1] = stroke_color[1]
    bg[:,:,2] = stroke_color[2]
    bg[:,:,3] = stroke_char_map

    fg_alpha = fg[:, :, 3] / 255.0
    bg_alpha = 1.0 - fg_alpha
    bg[y:y+h, x:x+w, :] = (fg_alpha[:, :, np.newaxis] * fg[:, :, :] + bg_alpha[:, :, np.newaxis] * bg[y:y+h, x:x+w, :])

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
    path = path.replace('\\', '/')
    if not font_cache.get(path):
        # To circumvent a bug with non ascii paths in windows use memory fonts
        # https://github.com/rougier/freetype-py/issues/157#issuecomment-1683713726
        font_cache[path] = freetype.Face(Path(path).open('rb'))
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

def calc_vertical(font_size: int, text: str, max_height: int):
    line_text_list = []
    # line_width_list = []
    line_height_list = []

    line_str = ""
    line_height = 0
    line_width_left = 0
    line_width_right = 0
    for i, cdpt in enumerate(text):
        if line_height == 0 and cdpt == ' ':
            continue
        cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
        ckpt = get_char_glyph(cdpt, font_size, 1)
        bitmap = ckpt.bitmap
        # spaces, etc
        if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
            char_offset_y = ckpt.metrics.vertBearingY >> 6
        else:
            char_offset_y = ckpt.metrics.vertAdvance >> 6
        char_width = bitmap.width
        char_bearing_x = ckpt.metrics.vertBearingX >> 6
        if line_height + char_offset_y > max_height:
            line_text_list.append(line_str)
            line_height_list.append(line_height)
            # line_width_list.append(line_width_left + line_width_right)
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
    # line_width_list.append(line_width_left + line_width_right)

    # box_calc_x = sum(line_width_list) + (len(line_width_list) - 1) * spacing_x
    # box_calc_y = max(line_height_list)
    return line_text_list, line_height_list

def put_char_vertical(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int):
    pen = pen_l.copy()

    is_pun = is_punctuation(cdpt)
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
        bitmap_border = np.array(bitmap_b.buffer, dtype = np.uint8).reshape(bitmap_b.rows, bitmap_b.width)
        canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width] = cv2.add(canvas_border[pen_border[1]:pen_border[1]+bitmap_b.rows, pen_border[0]:pen_border[0]+bitmap_b.width], bitmap_border)
    return char_offset_y

def put_text_vertical(font_size: int, text: str, h: int, alignment: str, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]], line_spacing: int):
    text = compact_special_symbols(text)
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_x = int(font_size * (line_spacing or 0.2))

    # make large canvas
    num_char_y = h // font_size
    num_char_x = len(text) // num_char_y + 1
    canvas_x = font_size * num_char_x + spacing_x * (num_char_x - 1) + (font_size + bg_size) * 2
    canvas_y = font_size * num_char_y + (font_size + bg_size) * 2
    line_text_list, line_height_list = calc_vertical(font_size, text, h)
    # print(line_text_list, line_height_list)

    canvas_text = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [canvas_text.shape[1] - (font_size + bg_size), font_size + bg_size]

    # write stuff
    for line_text, line_height in zip(line_text_list, line_height_list):
        pen_line = pen_orig.copy()
        if alignment == 'center':
            pen_line[1] += (max(line_height_list) - line_height) // 2
        elif alignment == 'right':
            pen_line[1] += max(line_height_list) - line_height

        for c in line_text:
            offset_y = put_char_vertical(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[1] += offset_y
        pen_orig[0] -= spacing_x + font_size

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, w, h = cv2.boundingRect(canvas_border)
    return line_box[y:y+h, x:x+w]

def select_hyphenator(lang: str):
    lang = standardize_tag(lang)
    if lang not in HYPHENATOR_LANGUAGES:
        for avail_lang in reversed(HYPHENATOR_LANGUAGES):
            if avail_lang.startswith(lang):
                lang = avail_lang
                break
        else:
            return None
    try:
        return Hyphenator(lang)
    except Exception:
        return None

# @functools.lru_cache(maxsize = 1024, typed = True)
def get_char_offset_x(font_size: int, cdpt: str):
    c, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 0)
    glyph = get_char_glyph(c, font_size, 0)
    bitmap = glyph.bitmap
    # Extract length
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        # spaces, etc
        char_offset_x = glyph.advance.x >> 6
    else:
        char_offset_x = glyph.metrics.horiAdvance >> 6
    return char_offset_x

def get_string_width(font_size: int, text: str):
    return sum([get_char_offset_x(font_size, c) for c in text])

def calc_horizontal(font_size: int, text: str, max_width: int, max_height: int, language: str = 'en_US', hyphenate: bool = True) -> Tuple[List[str], List[int]]:
    """
    Splits up a string of text into lines. Returns list of lines and their widths.
    Will go over max_height if too much text is present.
    """
    max_width = max(max_width, 2 * font_size)

    whitespace_offset_x = get_char_offset_x(font_size, ' ')
    hyphen_offset_x = get_char_offset_x(font_size, '-')

    # Split text into words and precalculate each word width
    words = re.split(r'\s+', text)
    word_widths = []
    for i, word in enumerate(words):
        word_widths.append(get_string_width(font_size, word))

    # Try to increase width usage if a height overflow is unavoidable
    while True:
        max_lines = max_height // font_size + 1
        expected_size = sum(word_widths) + max((len(word_widths) - 1) * whitespace_offset_x - (max_lines - 1) * hyphen_offset_x, 0)
        max_size = max_width * max_lines
        if max_size < expected_size:
            multiplier = np.sqrt(expected_size / max_size)
            max_width *= max(multiplier, 1.05)
            max_height *= multiplier
        else:
            break

    # Split words into syllables
    syllables = []
    hyphenator = select_hyphenator(language)
    for i, word in enumerate(words):
        new_syls = []
        if hyphenator and len(word) <= 100:
            try:
                new_syls = hyphenator.syllables(word)
            except Exception:
                new_syls = []
        if len(new_syls) == 0:
            if len(word) <= 3:
                new_syls = [word]
            else:
                new_syls = list(word)

        # # Make sure no syllable goes over max_width
        # for syl in syllables[-1]:
        #     w = get_string_width(font_size, syl)
        #     if w > max_width:
        #         max_width = w

        # Split up syllables that are too large
        normalized_syls = []
        for syl in new_syls:
            syl_width = get_string_width(font_size, syl)
            if syl_width > max_width:
                normalized_syls.extend(list(syl))
            else:
                normalized_syls.append(syl)
        syllables.append(normalized_syls)

    line_words_list = []
    line_width_list = []
    hyphenation_idx_list = []
    line_words = []
    line_width = 0
    hyphenation_idx = 0

    def break_line():
        nonlocal line_words, line_width, hyphenation_idx
        line_words_list.append(line_words)
        line_width_list.append(line_width)
        hyphenation_idx_list.append(hyphenation_idx)
        line_words = []
        line_width = 0
        hyphenation_idx = 0

    def get_present_syllables_range(line_idx, word_pos):
        while word_pos < 0:
            word_pos += len(line_words_list[line_idx])
        word_idx = line_words_list[line_idx][word_pos]
        syl_start_idx = 0
        syl_end_idx = len(syllables[word_idx])
        if line_idx > 0 and word_pos == 0 and line_words_list[line_idx - 1][-1] == word_idx:
            syl_start_idx = hyphenation_idx_list[line_idx - 1]
        if line_idx < len(line_words_list) - 1 and word_pos == len(line_words_list[line_idx]) - 1 \
            and line_words_list[line_idx + 1][0] == word_idx:
            syl_end_idx = hyphenation_idx_list[line_idx]
        return syl_start_idx, syl_end_idx

    def get_present_syllables(line_idx, word_pos):
        syl_start_idx, syl_end_idx = get_present_syllables_range(line_idx, word_pos)
        return syllables[line_words_list[line_idx][word_pos]][syl_start_idx:syl_end_idx]


    # Step 1:
    # Arrange words without hyphenating unless necessary

    i = 0
    while True:
        if i >= len(words):
            if line_width > 0:
                break_line()
            break

        current_width = whitespace_offset_x if line_width > 0 else 0

        if line_width + current_width + word_widths[i] <= max_width + hyphen_offset_x:
            line_words.append(i)
            line_width += current_width + word_widths[i]
            i += 1
        elif word_widths[i] > max_width:
            # We know no syllable can be larger than max_width
            j = 0
            hyphenation_idx = 0
            while j < len(syllables[i]):
                syl = syllables[i][j]
                syl_width = get_string_width(font_size, syl)
                if line_width + current_width + syl_width <= max_width:
                    current_width += syl_width
                    j += 1
                    hyphenation_idx = j
                else:
                    if hyphenation_idx > 0:
                        line_words.append(i)
                        line_width += current_width
                    current_width = 0
                    break_line()
            line_words.append(i)
            line_width += current_width
            i += 1
        else:
            break_line()


    # Step 2:
    # Compare two adjacent lines and try to hyphenate backwards

    # Avoid hyphenation if max_lines isn't fully used
    if hyphenate and len(line_words_list) > max_lines:
        line_idx = 0
        while line_idx < len(line_words_list) - 1:
            line_words1 = line_words_list[line_idx]
            line_words2 = line_words_list[line_idx + 1]
            left_space = max_width - line_width_list[line_idx]

            # Move syllables from below line to above
            first_word = True
            while len(line_words2) != 0:
                word_idx = line_words2[0]

                # A bit messy but were basically trying to only use the syllables on the current line
                if first_word and word_idx == line_words1[-1]:
                    syl_start_idx = hyphenation_idx_list[line_idx]
                    if line_idx < len(line_width_list) - 2 and word_idx == line_words_list[line_idx + 2][0]:
                        syl_end_idx = hyphenation_idx_list[line_idx + 1]
                    else:
                        syl_end_idx = len(syllables[word_idx])
                else:
                    left_space -= whitespace_offset_x
                    syl_start_idx = 0
                    syl_end_idx = len(syllables[word_idx]) if len(line_words2) > 1 else hyphenation_idx_list[line_idx + 1]
                first_word = False

                current_width = 0
                for i in range(syl_start_idx, syl_end_idx):
                    syl = syllables[word_idx][i]
                    syl_width = get_string_width(font_size, syl)
                    if left_space > current_width + syl_width:
                        current_width += syl_width
                    else:
                        # Splitting up word
                        if current_width > 0:
                            # We dont want very small splits
                            # if 
                            left_space -= current_width
                            line_width_list[line_idx] = max_width - left_space
                            hyphenation_idx_list[line_idx] = i
                            line_words1.append(word_idx)
                        break
                else:
                    # Whole word was brought to above line
                    left_space -= current_width
                    line_width_list[line_idx] = max_width - left_space
                    line_words1.append(word_idx)
                    line_words2.pop(0)
                    continue
                break

            if len(line_words2) == 0:
                line_words_list.pop(line_idx + 1)
                line_width_list.pop(line_idx + 1)
                hyphenation_idx_list.pop(line_idx)
            else:
                line_idx += 1

    
    # Step 3
    # Move single char syllables on the left up and those on the right down

    line_idx = 0
    while line_idx < len(line_words_list) - 1:
        line_words1 = line_words_list[line_idx]
        line_words2 = line_words_list[line_idx + 1]
        merged_word_idx = -1

        if line_words1[-1] == line_words2[0]:
            word1_text = ''.join(get_present_syllables(line_idx, -1))
            word2_text = ''.join(get_present_syllables(line_idx + 1, 0))
            word1_width = get_string_width(font_size, word1_text)
            word2_width = get_string_width(font_size, word2_text)
            if len(word2_text) == 1 or word2_width < font_size:
                merged_word_idx = line_words1[-1]
                line_words2.pop(0)
                line_width_list[line_idx] += word2_width
                line_width_list[line_idx + 1] -= word2_width + whitespace_offset_x
            elif len(word1_text) == 1 or word1_width < font_size:
                merged_word_idx = line_words1[-1]
                line_words1.pop(-1)
                line_width_list[line_idx] -= word1_width + whitespace_offset_x
                line_width_list[line_idx + 1] += word1_width

        if len(line_words1) == 0:
            line_words_list.pop(line_idx)
            line_width_list.pop(line_idx)
            hyphenation_idx_list.pop(line_idx)
        elif len(line_words2) == 0:
            line_words_list.pop(line_idx + 1)
            line_width_list.pop(line_idx + 1)
            hyphenation_idx_list.pop(line_idx)
        # We dont want all single letters to be merged
        elif line_idx >= len(line_words_list) - 1 or line_words_list[line_idx + 1] != merged_word_idx:
            line_idx += 1


    # Step 4
    # Assemble line_text_list

    use_hyphen_chars = hyphenate and hyphenator and max_width > 1.5 * font_size and len(words) > 1

    line_text_list = []
    for i, line in enumerate(line_words_list):
        line_text = ''
        for j, word_idx in enumerate(line):
            syl_start_idx, syl_end_idx = get_present_syllables_range(i, j)
            current_syllables = syllables[word_idx][syl_start_idx:syl_end_idx]
            line_text += ''.join(current_syllables)
            if len(line_text) == 0:
                continue
            if j == 0 and i > 0 and line_text_list[-1][-1] == '-' and line_text[0] == '-':
                line_text = line_text[1:]
                line_width_list[i] -= hyphen_offset_x
            if j < len(line) - 1 and len(line_text) > 0:
                line_text += ' '
            elif use_hyphen_chars and syl_end_idx != len(syllables[word_idx]) and len(words[word_idx]) > 3 and line_text[-1] != '-' \
                and not (syl_end_idx < len(syllables[word_idx]) and not re.search(r'\w', syllables[word_idx][syl_end_idx][0])):
                line_text += '-'
                # hyphen_offset was ignored in previous steps
                line_width_list[i] += hyphen_offset_x

        # print(line_text, get_string_width(font_size, line_text), line_width_list[i])
        # assert(line_width_list[i] == get_string_width(font_size, line_text))

        # Shouldn't be needed but there is apparently still a bug somewhere (See #458)
        line_width_list[i] = get_string_width(font_size, line_text)
        line_text_list.append(line_text)

    return line_text_list, line_width_list


def put_char_horizontal(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int):
    pen = pen_l.copy()

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

def put_text_horizontal(font_size: int, text: str, width: int, height: int, alignment: str,
                        reversed_direction: bool, fg: Tuple[int, int, int], bg: Tuple[int, int, int],
                        lang: str = 'en_US', hyphenate: bool = True, line_spacing: int = 0):
    text = compact_special_symbols(text)
    bg_size = int(max(font_size * 0.07, 1)) if bg is not None else 0
    spacing_y = int(font_size * (line_spacing or 0.01))

    # calc
    # print(width)
    line_text_list, line_width_list = calc_horizontal(font_size, text, width, height, lang, hyphenate)
    # print(line_text_list, line_width_list)

    # make large canvas
    canvas_w = max(line_width_list) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [font_size + bg_size, font_size + bg_size]
    if reversed_direction:
        # right to left languages have to be rendered in the correct order (starting from right)
        # so that the white outline of characters dont go over black parts of neighbouring characters
        pen_orig[0] = canvas_w - bg_size - 10

    # write stuff
    for line_text, line_width in zip(line_text_list, line_width_list):
        pen_line = pen_orig.copy()
        if alignment == 'center':
            pen_line[0] += (max(line_width_list) - line_width) // 2 * (-1 if reversed_direction else 1)
        elif alignment == 'right' and not reversed_direction:
            pen_line[0] += max(line_width_list) - line_width
        elif alignment == 'left' and reversed_direction:
            pen_line[0] -= max(line_width_list) - line_width
            pen_line[0] = max(line_width, pen_line[0])
        # print((line_width, pen_line[0], canvas_w))
        # print(0, pen_line, line_text)

        for c in line_text:
            if reversed_direction:
                cdpt, rot_degree = CJK_Compatibility_Forms_translate(c, 0)
                glyph = get_char_glyph(cdpt, font_size, 0)
                offset_x = glyph.metrics.horiAdvance >> 6
                pen_line[0] -= offset_x
            # print(1, pen_line, c)
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            if not reversed_direction:
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
