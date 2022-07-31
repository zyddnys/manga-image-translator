import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import List, Union, Tuple

from textblockdetector import TextBlock
from utils import Quadrilateral


class Line:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) -> List:
    assert ratio > 1.0
    
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    # https://numpy.org/doc/stable/reference/generated/numpy.roots.html
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2 ))
    delta_w = int(delta * aspect_ratio)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[[0, 2]] = np.clip(rect[[0, 2]], 0, im_w)
    rect[[1, 3]] = np.clip(rect[[1, 3]], 0, im_h)
    return rect.tolist()

def extract_ballon_region(img: np.ndarray, ballon_rect: List, show_process=False, enlarge_ratio=2.0) -> Tuple[np.ndarray, int, List]:

    x1, y1, x2, y2 = ballon_rect[0], ballon_rect[1], \
        ballon_rect[2] + ballon_rect[0], ballon_rect[3] + ballon_rect[1]
    if enlarge_ratio > 1:
        x1, y1, x2, y2 = enlarge_window([x1, y1, x2, y2], img.shape[1], img.shape[0], enlarge_ratio, aspect_ratio=ballon_rect[3] / ballon_rect[2])

    img = img[y1:y2, x1:x2].copy()

    kernel = np.ones((3,3),np.uint8)
    orih, oriw = img.shape[0], img.shape[1]
    scaleR = 1
    if orih > 300 and oriw > 300:
        scaleR = 0.6
    elif orih < 120 or oriw < 120:
        scaleR = 1.4

    if scaleR != 1:
        h, w = img.shape[0], img.shape[1]
        orimg = np.copy(img)
        img = cv2.resize(img, (int(w*scaleR), int(h*scaleR)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[0], img.shape[1]
    img_area = h * w

    cpimg = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(cpimg, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)

    cons, hiers = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), BLACK, 1, cv2.LINE_8)

    ballon_mask, outer_index = np.zeros((h, w), np.uint8), -1

    min_retval = np.inf
    mask = np.zeros((h, w), np.uint8)
    difres = 10
    seedpnt = (int(w/2), int(h/2))
    for ii in range(len(cons)):
        rect = cv2.boundingRect(cons[ii])
        if rect[2]*rect[3] < img_area*0.4:
            continue
        
        mask = cv2.drawContours(mask, cons, ii, (255), 2)
        cpmask = np.copy(mask)
        cv2.rectangle(mask, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
        retval, _, _, rect = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))

        if retval <= img_area * 0.3:
            mask = cv2.drawContours(mask, cons, ii, (0), 2)
        if retval < min_retval and retval > img_area * 0.3:
            min_retval = retval
            ballon_mask = cpmask

    ballon_mask = 127 - ballon_mask
    ballon_mask = cv2.dilate(ballon_mask, kernel,iterations = 1)
    ballon_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    box_kernel = int(np.sqrt(ballon_area) / 30)
    if box_kernel > 1:
        box_kernel = np.ones((box_kernel,box_kernel),np.uint8)
        ballon_mask = cv2.dilate(ballon_mask, box_kernel, iterations = 1)

    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))

    if show_process:
        cv2.imshow('ballon_mask', ballon_mask)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return ballon_mask, (ballon_mask > 0).sum(), [x1, y1, x2, y2]

def render_lines(
    line_lst: List[Line], 
    canvas_h: int, 
    canvas_w: int, 
    font: ImageFont.FreeTypeFont, 
    stroke_width: int, 
    font_color: Tuple[int] = (0, 0, 0), 
    stroke_color: Tuple[int] = (255, 255, 255)) -> Image.Image:

    c = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
    d = ImageDraw.Draw(c)
    d.fontmode = 'L'
    for line in line_lst:
        d.text((line.pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=stroke_width, stroke_fill=stroke_color)
    return c


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

def render_textblock_list_eng(img: np.ndarray, blk_list: List[TextBlock], font_path: str, scale_quality=1.0, align_center=True, size_tol=1.0, font_size_offset: int = 0):
    pilimg = Image.fromarray(img)
    for blk in blk_list:
        if blk.vertical:
            blk.angle -= 90
        sw_r = 0.1
        fs = int(blk.font_size / (1 + 2*sw_r) * scale_quality) + font_size_offset
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