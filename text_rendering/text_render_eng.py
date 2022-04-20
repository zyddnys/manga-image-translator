from PIL import ImageFont, ImageDraw, Image
import numpy as np
from typing import List, Union
from textblockdetector import TextBlock

from utils import Quadrilateral


class Line:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)


def text_to_word_list(text: str) -> List[str]:
    text = text.upper().replace('  ', ' ')
    processed_text = ''

    # dumb way to insure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
            if c in ['.', '?', '!'] and ii < text_len - 1:
                next_c = text[ii + 1]
                if next_c.isalpha() or next_c.isnumeric():
                    processed_text += c + ' '
                else:
                    processed_text += c
            else:
                processed_text += c
    word_list = processed_text.split(' ')
    words = []
    skip_next = False
    word_num = len(word_list)
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if ii < word_num - 1:
            if len(word) == 1 or len(word_list[ii + 1]) == 1:
                skip_next = True
                word = word + ' ' + word_list[ii + 1]
        words.append(word)
    return words

def render_textblock_list_eng(img: np.ndarray, blk_list: List[TextBlock], font_path: str, scale_quality=1.0, align_center=True, size_tol=1.0):
    pilimg = Image.fromarray(img)
    for blk in blk_list:
        if blk.vertical:
            blk.angle -= 90
        sw_r = 0.1
        fs = int(blk.font_size / (1 + 2*sw_r) * scale_quality) 
        min_bbox = blk.min_rect(rotate_back=False)[0]
        bx, by = min_bbox[0]
        bw, bh = min_bbox[2] - min_bbox[0]
        cx, cy = bx + bw / 2, by + bh / 2
        bw = bw * scale_quality

        font = ImageFont.truetype(font_path, fs)
        words = text_to_word_list(blk.translation)
        if not len(words):
            continue

        base_length = -1
        w_list = []
        
        sw = int(sw_r * font.size)
        line_height = int((1 + 2*sw_r) * font.getmetrics()[0])

        for word in words:
            wl = font.getlength(word)
            w_list.append(wl)
            if wl > base_length:
                base_length = wl
        base_length = max(base_length, bw)
        space_l = font.getlength(' ')
        pos_x, pos_y = 0, 0
        line = Line(words[0], 0, 0, w_list[0])
        line_lst = [line]
        for word, wl in zip(words[1:], w_list[1:]):
            added_len = int(space_l + wl + line.length)
            if added_len > base_length:
                pos_y += line_height
                line = Line(word, 0, pos_y, wl)
                line_lst.append(line)
            else:
                line.text = line.text + ' ' + word
                line.length = added_len
        last_line = line_lst[-1]
        canvas_h = last_line.pos_y + line_height
        canvas_w = int(base_length)

        font_color = (0, 0, 0)
        stroke_color = (255, 255, 255)
        img = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.fontmode = 'L'

        for line in line_lst:
            pos_x = int((base_length - line.length) / 2) if align_center else 0
            d.text((pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=sw, stroke_fill=stroke_color)
        
        if abs(blk.angle) > 3:
            img = img.rotate(-blk.angle, expand=True)
        im_w, im_h = img.size
        scale = min(bh / im_h * size_tol, bw / im_w * size_tol)
        if scale < 1:
            img = img.resize((int(im_w*scale), int(im_h*scale)))

        im_w, im_h = img.size
        paste_x, paste_y = int(cx - im_w / 2), int(cy - im_h / 2)
        pilimg.paste(img, (paste_x, paste_y), mask=img)
    
    return np.array(pilimg)