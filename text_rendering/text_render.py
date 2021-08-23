
from itertools import filterfalse
import pickle
from typing import List, Tuple, Optional

import numpy as np

import cv2
import unicodedata
import freetype
from utils import BBox, Quadrilateral

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
	"""Checks whether `chars` is a whitespace character."""
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
	"""Checks whether `chars` is a whitespace character."""
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

AVAILABLE_FONTS =[]
FONT_FACE_MAP = {}
CDPT_FONT_MAP = {}

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

def CJK_Compatibility_Forms_translate(cdpt: str, direction: int) :
	if cdpt == 'ー' and direction == 1 :
		return 'ー', 90
	if cdpt in ["︰", "︱", "︲", "︳", "︴", "︵", "︶", "︷", "︸", "︹", "︺", "︻", "︼", "︽", "︾", "︿", "﹀", "﹁", "﹂", "﹃", "﹄", "﹅", "﹆", "﹇", "﹈", "﹉", "﹊", "﹋", "﹌", "﹍", "﹎", "﹏", "⋮"] :
		if direction == 0 :
			# translate
			return CJK_V2H[cdpt], 0
		else :
			return cdpt
	elif cdpt in ["‥", "—", "–", "_", "_", "(", ")", "（", "）", "{", "}", "〔", "〕", "【", "】", "《", "》", "〈", "〉", "「", "」", "『", "』", "﹑", "﹆", "[", "]", "﹉", "﹊", "﹋", "﹌", "﹍", "﹎", "﹏", "…"] :
		if direction == 1 :
			# translate
			return CJK_H2V[cdpt], 0
		else :
			return cdpt, 0
	return cdpt, 0

def rotate_image(image, angle) :
	if angle == 0 :
		return image, (0, 0)
	image_exp = np.zeros((round(image.shape[0] * 1.5), round(image.shape[1] * 1.5), image.shape[2]), dtype = np.uint8)
	diff_i = (image_exp.shape[0] - image.shape[0]) // 2
	diff_j = (image_exp.shape[1] - image.shape[1]) // 2
	image_exp[diff_i:diff_i+image.shape[0], diff_j:diff_j+image.shape[1]] = image
	# from https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
	image_center = tuple(np.array(image_exp.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image_exp, rot_mat, image_exp.shape[1::-1], flags=cv2.INTER_LINEAR)
	if angle == 90 :
		return result, (0, 0)
	return result, (diff_i, diff_j)

def add_color(bw_char_map, color, border_color = None, border_size: int = 0) :
	fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 3), dtype = np.uint8)
	if bw_char_map.size == 0 :
		return fg.astype(np.uint8), bw_char_map, color, border_color if border_size > 0 else None
	color_np = np.array(color, dtype = np.uint8)
	if border_color and border_size > 0 :
		bg_color_np = np.array(border_color, dtype = np.uint8)
		fg[:] = bg_color_np
		alpha = cv2.dilate(bw_char_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size * 2 + 1, border_size * 2 + 1)))
		fg[bw_char_map > 127] = color_np
		return fg.astype(np.uint8), alpha, color, border_color if border_size > 0 else None
	else :
		fg[:] = color_np
		return fg.astype(np.uint8), bw_char_map, color, border_color if border_size > 0 else None

CACHED_FONT_FACE = []

import functools
import copy

class namespace :
	pass

class Glyph :
	def __init__(self, glyph) :
		self.bitmap = namespace()
		self.bitmap.buffer = glyph.bitmap.buffer
		self.bitmap.rows = glyph.bitmap.rows
		self.bitmap.width = glyph.bitmap.width
		self.advance = namespace()
		self.advance.x = glyph.advance.x
		self.advance.y = glyph.advance.y
		self.bitmap_left = glyph.bitmap_left
		self.bitmap_top = glyph.bitmap_top

@functools.lru_cache(maxsize = 1024, typed = True)
def get_char_glyph(cdpt, font_size: int, direction: int) :
	global CACHED_FONT_FACE
	for i, face in enumerate(CACHED_FONT_FACE) :
		if face.get_char_index(cdpt) == 0 and i != len(CACHED_FONT_FACE) - 1 :
			continue
		if direction == 0 :
			face.set_pixel_sizes( 0, font_size )
		elif direction == 1 :
			face.set_pixel_sizes( font_size, 0 )
		face.load_char(cdpt)
		return Glyph(face.glyph), face.glyph.bitmap.rows * face.glyph.bitmap.width == 0

def put_char(canvas: np.ndarray, mask: np.ndarray, x: int, y: int, font_size: int, rot: int, cdpt: str, direction: int, char_color = (0,0,0), border_color = (0,255,0), border_size = 2) :
	is_pun = _is_punctuation(cdpt)
	cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, direction)
	old_font_size = font_size
	font_size += border_size * 2
	x -= border_size
	y -= border_size
	glyph, empty_char = get_char_glyph(cdpt, old_font_size, direction)
	offset_x = glyph.advance.x>>6
	offset_y = glyph.advance.y>>6
	bitmap = glyph.bitmap
	if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width :
		if offset_y == 0 and direction == 1 :
			offset_y = offset_x
		return offset_x, offset_y

	char_map = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
	size = font_size
	if is_pun and direction == 1 :
		x2, y2, w2, h2 = cv2.boundingRect(char_map.astype(np.uint8) * 255)
		if w2 > font_size * 0.7 :
			place_x = glyph.bitmap_left
		else :
			place_x = (font_size - w2) // 2
		if h2 > font_size * 0.7 :
			place_y = max(3 * font_size // 4 - glyph.bitmap_top, 0)
		else :
			place_y = (font_size - h2) // 2
	else :
		place_x = glyph.bitmap_left#max((offset_x - bitmap.width) >> 1, 0)
		place_y = max(3 * font_size // 4 - glyph.bitmap_top, 0)
	place_x = max(place_x, 0)
	place_y = max(place_y, 0)
	new_char_map = np.zeros((size + place_y, size + place_x),dtype=np.uint8)
	available_region = new_char_map[place_y:place_y+char_map.shape[0], place_x:place_x+char_map.shape[1]]
	new_char_map[place_y:place_y+char_map.shape[0], place_x:place_x+char_map.shape[1]] = char_map[:available_region.shape[0],:available_region.shape[1]]
	char_map = new_char_map
	char_map = char_map.reshape((char_map.shape[0],char_map.shape[1],1))
	available_shape = canvas[y :y+font_size,x: x+font_size,:].shape
	char_map = char_map[:available_shape[0],:available_shape[1]]
	if len(char_map.shape) == 3 :
		char_map = char_map.squeeze(-1)
	if border_color :
		char_map, char_map_alpha, char_color, border_color = add_color(char_map, char_color, border_color=border_color, border_size=border_size)
	else :
		char_map, char_map_alpha, char_color, border_color = add_color(char_map, char_color)
	canvas[y:y+font_size,x: x+font_size,:] = char_map
	mask[y:y+font_size,x: x+font_size] += char_map_alpha
	if offset_y == 0 and direction == 1 :
		offset_y = old_font_size
	return offset_x, offset_y

def put_text_vertical(font_size: int, mag_ratio: float, img: np.ndarray, mask: np.ndarray, text: str, line_count: int, lines: List[Quadrilateral], x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	x1 = x
	x2 = x + w
	y1 = y
	y2 = y + h
	#cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	# font_size = round(w / (line_count * 0.9))
	rows = h // font_size
	cols = w // font_size
	# while rows * cols < len(text) :
	# 	font_size -= 1
	# 	rows = h // font_size
	# 	cols = w // font_size
	fg_avg = (fg[0] + fg[1] + fg[2]) / 3
	if bg :
		bg_avg = (bg[0] + bg[1] + bg[2]) / 3
		if abs(fg_avg - bg_avg) < 40 :
			bg = None
	bgsize = int(max(font_size * 0.07, 1)) if bg else 0
	spacing_y = 0#int(max(font_size * 0.05, 1))
	spacing_x = spacing_y
	x = x2 - spacing_x - font_size
	y = y1 + max(spacing_y, 0)
	txt_i = 0
	rot = 0
	j = 0
	while True :
		new_length = rows
		if not new_length :
			continue
		y = y1 + spacing_y
		cur_line_bbox = lines[j] if j < len(lines) else BBox(0, y1, 0, h, '', 0)
		while True :
			x_offset, y_offset = put_char(img, mask, x, y, font_size, rot, text[txt_i], 1, char_color=fg,border_color=bg,border_size=bgsize)
			txt_i += 1
			if txt_i >= len(text) :
				return True
			y += spacing_y + y_offset
			if y + font_size > y2 :
				break
			if y > cur_line_bbox.height() * mag_ratio + y1 + font_size * 2 and j + 1 < len(lines) :
				break
		x -= spacing_x + font_size
		j += 1
	return True

def put_text_horizontal(font_size: int, mag_ratio: float, img: np.ndarray, mask: np.ndarray, text: str, line_count: int, lines: List[Quadrilateral], x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	x1 = x
	x2 = x + w
	y1 = y
	y2 = y + h
	#cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	# font_size = round(h / (line_count * 0.9))
	rows = h // font_size
	cols = w // font_size
	# while rows * cols < len(text) :
	# 	font_size -= 1
	# 	rows = h // font_size
	# 	cols = w // font_size
	fg_avg = (fg[0] + fg[1] + fg[2]) / 3
	if bg :
		bg_avg = (bg[0] + bg[1] + bg[2]) / 3
		if abs(fg_avg - bg_avg) < 40 :
			bg = None
	bgsize = int(max(font_size * 0.07, 1)) if bg else 0
	spacing_x = 0#int(max(font_size * 0.05, 1))
	spacing_y = spacing_x
	x = x1 + max(spacing_x, 0)
	y = y1 + spacing_y
	txt_i = 0
	rot = 0
	i = 0
	while True :
		new_length = cols
		if not new_length :
			continue
		x = x1 + spacing_x
		cur_line_bbox = lines[i] if i < len(lines) else BBox(x1, 0, w, 0, '', 0)
		while True :
			x_offset, y_offset = put_char(img, mask, x, y, font_size, rot, text[txt_i], 0, char_color=fg,border_color=bg,border_size=bgsize)
			txt_i += 1
			if txt_i >= len(text) :
				return True
			x += spacing_x + x_offset
			if x + font_size > x2 :
				break
			if x > cur_line_bbox.width() * mag_ratio + x1 + font_size * 2 and i + 1 < len(lines) :
				break
		y += font_size + spacing_y
		i += 1
	return True

def put_text(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	pass

def prepare_renderer(font_filenames = ['fonts/Arial-Unicode-Regular.ttf', 'fonts/msyh.ttc', 'fonts/msgothic.ttc']) :
	global CACHED_FONT_FACE
	for font_filename in font_filenames :
		CACHED_FONT_FACE.append(freetype.Face(font_filename))

def test() :
	prepare_renderer()
	canvas = np.ones((4096, 2590, 3), dtype = np.uint8) * 255
	put_text_vertical(canvas, '《因为不同‼ [这"真的是普]通的》肉！那个“姑娘”的恶作剧！是吗？咲夜⁉', 4, [], 2143, 3219, 355, 830, (0, 0, 0), None)
	put_text_horizontal(canvas, '“添加幽默”FOR if else !?xxj', 1, [], 242, 87, 2093, 221, (0, 0, 0), None)
	cv2.imwrite('text_render_combined.png', canvas)

if __name__ == '__main__' :
	test()
