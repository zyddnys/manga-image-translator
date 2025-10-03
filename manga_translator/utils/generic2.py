# This file are functions that are essential for the renderer export

import unicodedata
from typing import List

import cv2
import numpy as np


def color_difference(rgb1: List, rgb2: List) -> float:
    # https://en.wikipedia.org/wiki/Color_difference#CIE76
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(
        np.float32)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2)
    return diff.item()


def is_punctuation(ch):
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


def is_whitespace(ch):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False


def is_control(ch):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_valuable_char(ch):
    # return re.search(r'[^\d\W]', ch)
    return not is_punctuation(ch) and not is_control(ch) and not is_whitespace(ch) and not ch.isdigit()


def is_valuable_text(text):
    for ch in text:
        if is_valuable_char(ch):
            return True
    return False


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


def is_right_to_left_char(ch):
    """Checks whether the char belongs to a right to left alphabet."""
    # Arabic (from https://stackoverflow.com/a/49346768)
    if ('\u0600' <= ch <= '\u06FF' or
            '\u0750' <= ch <= '\u077F' or
            '\u08A0' <= ch <= '\u08FF' or
            '\uFB50' <= ch <= '\uFDFF' or
            '\uFE70' <= ch <= '\uFEFF' or
            '\U00010E60' <= ch <= '\U00010E7F' or
            '\U0001EE00' <= ch <= '\U0001EEFF'):
        return True
    return False
