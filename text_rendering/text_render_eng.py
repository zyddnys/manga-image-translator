import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

from .text_render import get_char_glyph, put_char_horizontal, add_color
from detection.ctd_utils import TextBlock
from .ballon_extractor import extract_ballon_region

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', "\""}


class Textline:
    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def render_lines(
    textlines: List[Textline],
    canvas_h: int,
    canvas_w: int,
    font_size: int,
    stroke_width: int,
    fg: Tuple[int] = (0, 0, 0),
    bg: Tuple[int] = (255, 255, 255)) -> Image.Image:

    # bg_size = int(max(font_size * 0.1, 1)) if bg is not None else 0
    bg_size = stroke_width
    spacing_y = int(font_size * 0.01)

    # make large canvas
    canvas_w = max([l.length for l in textlines]) + (font_size + bg_size) * 2
    canvas_h = font_size * len(textlines) + spacing_y * (len(textlines) - 1)  + (font_size + bg_size) * 2
    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()

    # pen (x, y)
    pen_orig = [font_size + bg_size, font_size + bg_size]

    # write stuff
    for line in textlines:
        pen_line = pen_orig.copy()
        pen_line[0] += line.pos_x # center
        for c in line.text:
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size)
            pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    # colorize
    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)

    # rect
    x, y, width, height = cv2.boundingRect(canvas_border)
    return Image.fromarray(line_box[y:y+height, x:x+width])

    # c = Image.new('RGBA', (canvas_w, canvas_h), color = (0, 0, 0, 0))
    # d = ImageDraw.Draw(c)
    # d.fontmode = 'L'
    # for line in lines:
    #     d.text((line.pos_x, line.pos_y), line.text, font=font, fill=font_color, stroke_width=font_size, stroke_fill=stroke_color)
    # return c

def seg_eng(text: str) -> List[str]:
    """
    Extracts every word from text parameter
    """
    # TODO: replace with regexes

    text = text.upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to ensure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
        if c in PUNSET_RIGHT_ENG and ii < text_len - 1:
            next_c = text[ii + 1]
            if next_c.isalpha() or next_c.isnumeric():
                processed_text += c + ' '
            else:
                processed_text += c
        else:
            processed_text += c

    word_list = processed_text.split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if len(word) < 3:
            append_left, append_right = False, False
            len_word, len_next, len_prev = len(word), -1, -1
            if ii < word_num - 1:
                len_next = len(word_list[ii + 1])
            if ii > 0:
                len_prev = len(words[-1])
            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1
            if len_next > 0 and len_prev > 0:
                if len_next < len_prev:
                    append_right = cond_next
                else:
                    append_left = cond_prev
            elif len_next > 0:
                append_right = cond_next
            elif len_prev:
                append_left = cond_prev

            if append_left:
                words[-1] = words[-1] + ' ' + word
            elif append_right:
                words.append(word + ' ' + word_list[ii + 1])
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)
    return words

def layout_lines_aligncenter(
    mask: np.ndarray, 
    words: List[str], 
    word_lengths: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False)->List[Textline]:

    m = cv2.moments(mask)
    mask = 255 - mask
    centroid_y = int(m['m01'] / m['m00'])
    centroid_x = int(m['m10'] / m['m00'])

    # layout the central line, the center word is approximately aligned with the centroid of the mask
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    if num_words > 1:
        wl_array = np.array(word_lengths, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = np.argmin(np.abs(wl_cumsums))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = word_lengths[:central_index]
            sum_left = np.sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = word_lengths[central_index + 1:]
            sum_right = np.sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - word_lengths[central_index] // 2

    bh, bw = mask.shape[:2]
    central_line = Textline(words[central_index], pos_x, pos_y, word_lengths[central_index], spacing)
    line_bottom = pos_y + line_height
    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                if mask[pos_y: line_bottom, new_x_l].sum()==0 and mask[pos_y: line_bottom, new_r_l].sum() == 0:
                    left_valid = True
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if (new_x_r > 0 and new_r_r < bw):
                if mask[pos_y: line_bottom, new_x_r].sum()==0 and mask[pos_y: line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        insert_left = False
        if left_valid and right_valid:
            if sum_left > sum_right:
                insert_left = True
        elif left_valid:
            insert_left = True
        elif not right_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r
        if central_line.length > max_central_width:
            break

    central_line.strip_spacing()
    lines = [central_line]

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_right > 0:
                        w, wl = wlst_right.pop(0), len_right.pop(0)
                        sum_right -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        line_bottom = pos_y + line_height
        line = Textline(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                if new_len > max_central_width:
                    line_valid = False
                    if sum_left > 0:
                        w, wl = wlst_left.pop(-1), len_left.pop(-1)
                        sum_left -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line.strip_spacing()
                line = Textline(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
    # for line in lines:
    #     cv2.rectangle(rbgmsk, (line.pos_x, line.pos_y), (line.pos_x + line.length, line.pos_y + line_height), (0, 255, 0))
    # cv2.imshow('mask', rbgmsk)
    # cv2.waitKey(0)

    return lines

def render_textblock_list_eng(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_color = (0, 0, 0),
    stroke_color = (255, 255, 255),
    delimiter: str = ' ',
    align_center=True,
    line_spacing: float = 1.0,
    stroke_width: float = 0.1,
    size_tol: float = 1.0,
    ballonarea_thresh: float = 2,
    downscale_constraint: float = 0.7,
    ref_textballon: bool = True,
    original_img: np.ndarray = None,
) -> np.ndarray:

    r"""
    Args:
        downscale_constraint (float, optional): minimum scaling down ratio, prevent rendered text from being too small
        ref_textballon (bool, optional): take text balloons as reference for text layout. 
        original_img (np.ndarray, optional): original image used to extract text balloons.
    """

    def calculate_font_values(font_size: int, words: List[str]):
        font_size = int(font_size)
        sw = int(font_size * stroke_width)
        line_height = int(font_size * 0.8)
        delimiter_glyph = get_char_glyph(delimiter, font_size, 0)
        delimiter_len = delimiter_glyph.advance.x >> 6
        base_length = -1
        word_lengths = []
        for word in words:
            word_length = 0
            for cdpt in word:
                glyph = get_char_glyph(cdpt, font_size, 0)
                char_offset_x = glyph.metrics.horiAdvance >> 6
                word_length += char_offset_x
            word_lengths.append(word_length)
            if word_length > base_length:
                base_length = word_length
        return font_size, sw, line_height, delimiter_len, base_length, word_lengths

    pilimg = Image.fromarray(img)

    for region in text_regions:
        words = seg_eng(region.translation)
        if not words:
            continue

        font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(region.font_size, words)

        if ref_textballon:
            assert original_img is not None
            bounding_rect = region.bounding_rect()
            # non-dl textballon segmentation
            enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, 3)
            ballon_mask, ballon_area, xyxy = extract_ballon_region(original_img, bounding_rect, enlarge_ratio=enlarge_ratio)
            rotated, rx, ry = False, 0, 0
            if abs(region.angle) > 3:
                rotated = True
                region_angle_rad = np.deg2rad(region.angle)
                region_angle_sin = np.sin(region_angle_rad)
                region_angle_cos = np.cos(region_angle_rad)
                rotated_ballon_mask = Image.fromarray(ballon_mask).rotate(region.angle, expand=True)
                rotated_ballon_mask = np.array(rotated_ballon_mask)

                region.angle %= 360
                if region.angle > 0 and region.angle <= 90:
                    ry = abs(ballon_mask.shape[1] * region_angle_sin)
                elif region.angle > 90 and region.angle <= 180:
                    rx = abs(ballon_mask.shape[1] * region_angle_cos)
                    ry = rotated_ballon_mask.shape[0]
                elif region.angle > 180 and region.angle <= 270:
                    ry = abs(ballon_mask.shape[0] * region_angle_cos)
                    rx = rotated_ballon_mask.shape[1]
                else:
                    rx = abs(ballon_mask.shape[0] * region_angle_sin)
                ballon_mask = rotated_ballon_mask

            line_width = sum(word_lengths) + delimiter_len * (len(word_lengths) - 1)
            line_area = line_width * line_height + delimiter_len * (len(words) - 1) * line_height
            area_ratio = ballon_area / line_area
            resize_ratio = 1
            # if ballon_area is smaller than 2*line_area
            if area_ratio < ballonarea_thresh:
                # resize so that it is 2*line_area
                resize_ratio = ballonarea_thresh / area_ratio
                ballon_area = int(resize_ratio * ballon_area) # = ballonarea_thresh * line_area
                resize_ratio = min(np.sqrt(resize_ratio), (1/downscale_constraint)**2)
                rx *= resize_ratio
                ry *= resize_ratio
                ballon_mask = cv2.resize(ballon_mask, (int(resize_ratio * ballon_mask.shape[1]), int(resize_ratio * ballon_mask.shape[0])))

            # new region bbox
            region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(ballon_mask))

            font_size_multiplier = max(region_w / (base_length + 2*sw), downscale_constraint)
            if font_size_multiplier < 1:
                font_size = int(font_size * font_size_multiplier)
                font_size, sw, line_height, delimiter_len, base_length, word_lengths = calculate_font_values(font_size * font_size_multiplier, words)

            textlines = layout_lines_aligncenter(ballon_mask, words, word_lengths, delimiter_len, line_height, delimiter=delimiter)

            line_cy = np.array([line.pos_y for line in textlines]).mean() + line_height / 2
            region_cy = region_y + region_h / 2
            y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

            lines_x1, lines_x2 = [], []
            for line in textlines:
                lines_x1.append(line.pos_x)
                lines_x2.append(max(line.pos_x, 0) + line.length)
            lines_x1 = np.array(lines_x1)
            lines_x2 = np.array(lines_x2)
            canvas_x1, canvas_x2 = lines_x1.min() - sw, lines_x2.max() + sw
            canvas_y1, canvas_y2 = textlines[0].pos_y - sw, textlines[-1].pos_y + line_height + sw
            canvas_h = int(canvas_y2 - canvas_y1)
            canvas_w = int(canvas_x2 - canvas_x1)
            lines_map = np.zeros_like(ballon_mask, dtype=np.uint8)
            for line in textlines:
                # line.pos_y += y_offset
                cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset), (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)
                line.pos_x -= canvas_x1
                line.pos_y -= canvas_y1

            textlines_image = render_lines(textlines, canvas_h, canvas_w, font_size, sw, font_color, stroke_color)
            rel_cx = ((canvas_x1 + canvas_x2) / 2 - rx) / resize_ratio
            rel_cy = ((canvas_y1 + canvas_y2) / 2 - ry + y_offset) / resize_ratio

            lines_area = np.sum(lines_map)
            lines_area += (max(0, region_y - canvas_y1) + max(0, canvas_y2 - region_h - region_y)) * canvas_w * 255 \
                          + (max(0, region_x - canvas_x1) + max(0, canvas_x2 - region_w - region_x)) * canvas_h * 255

            valid_lines_ratio = lines_area / np.sum(cv2.bitwise_and(lines_map, ballon_mask))
            if valid_lines_ratio > 1: # text bbox > ballon area
                resize_ratio = min(resize_ratio * valid_lines_ratio, (1 / downscale_constraint) ** 2)

            if rotated:
                rcx = rel_cx * region_angle_cos - rel_cy * region_angle_sin
                rcy = rel_cx * region_angle_sin + rel_cy * region_angle_cos
                rel_cx = rcx
                rel_cy = rcy
                textlines_image = textlines_image.rotate(-region.angle, expand=True, resample=Image.BILINEAR)
                textlines_image = textlines_image.crop(textlines_image.getbbox())

            abs_cx = rel_cx + xyxy[0]
            abs_cy = rel_cy + xyxy[1]

            if resize_ratio != 1:
                textlines_image = textlines_image.resize((int(textlines_image.width / resize_ratio), int(textlines_image.height / resize_ratio)))
            abs_x = int(abs_cx - textlines_image.width / 2)
            abs_y = int(abs_cy - textlines_image.height / 2)
            pilimg.paste(textlines_image, (abs_x, abs_y), mask=textlines_image)
            # cv2.imshow('ballon_region', ballon_region)
            # cv2.imshow('cropped', original_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
            # cv2.imshow('raw_lines', np.array(raw_lines))
            # cv2.waitKey(0)

        # else: # older method
        #     min_bbox = region.min_rect(rotate_back=False)[0]
        #     bx, by = min_bbox[0]
        #     bw, bh = min_bbox[2] - min_bbox[0]
        #     cx, cy = bx + bw / 2, by + bh / 2
        #     base_length = max(base_length, bw)

        #     pos_x, pos_y = 0, 0
        #     line = Textline(words[0], 0, 0, wordlengths[0])
        #     lines = [line]
        #     for word, wl in zip(words[1:], wordlengths[1:]):
        #         added_len = int(delimiter_len + wl + line.length)
        #         if added_len > base_length:
        #             pos_y += line_height
        #             line = Textline(word, 0, pos_y, wl)
        #             lines.append(line)
        #         else:
        #             line.text = line.text + ' ' + word
        #             line.length = added_len
        #     last_line = lines[-1]
        #     canvas_h = last_line.pos_y + line_height
        #     canvas_w = int(base_length)

        #     for line in lines:
        #         line.pos_x = int((base_length - line.length) / 2) if align_center else 0
        #     textlines_image = render_lines(lines, canvas_h, canvas_w, font, sw, font_color, stroke_color)

        #     if abs(region.angle) > 3:
        #         textlines_image = textlines_image.rotate(-region.angle, expand=True)
        #     im_w, im_h = textlines_image.size
        #     scale = max(min(bh / im_h * size_tol, bw / im_w * size_tol), downscale_constraint)
        #     if scale < 1:
        #         textlines_image = textlines_image.resize((int(im_w*scale), int(im_h*scale)))

        #     im_w, im_h = textlines_image.size
        #     paste_x, paste_y = int(cx - im_w / 2), int(cy - im_h / 2)

        #     pilimg.paste(textlines_image, (paste_x, paste_y), mask=textlines_image)

    return np.array(pilimg)
