
from itertools import filterfalse
import pickle
from sys import flags
from typing import List, Tuple, Optional


import numpy as np

import cv2
import unicodedata

from time import sleep
from PIL import Image, ImageFont, ImageDraw
import math

if __name__ == '__main__' :
	import sys, os 
	p = os.path.abspath('.')
	sys.path.insert(1, p)

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
		self.slot = glyph
		self.bitmap = namespace()
		self.bitmap.buffer = glyph.bitmap.buffer
		self.bitmap.rows = glyph.bitmap.rows
		self.bitmap.width = glyph.bitmap.width
		self.advance = namespace()
		self.advance.x = glyph.advance.x
		self.advance.y = glyph.advance.y
		self.bitmap_left = glyph.bitmap_left
		self.bitmap_top = glyph.bitmap_top
	def get_glyph(self):
		return self.slot.get_glyph()



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

@functools.lru_cache(maxsize = 1024, typed = True)
def get_font(font_size: int, direction=0) :
	global CACHED_FONT_FACE
	for i, face in enumerate(CACHED_FONT_FACE) :
		return ImageFont.truetype(face, font_size)

def wrap_text(text, boxWidth, font, draw):
	textArr = text.split(" ")
	newStr = ""
	line = ""
	for i, c in enumerate(textArr):
			if i == 0:
					line += c
					if i == len(textArr) - 1:
							newStr += line
			else:
					if draw.textlength(line + " " + c, font=font) > boxWidth:
							newStr += line + "\n"
							line = c
							if i == len(textArr) - 1:
									newStr += c
					else:
							line += " " + c
							if i == len(textArr) - 1:
									newStr += line
	return newStr
	

def put_char(canvas: np.ndarray, mask: np.ndarray, x: int, y: int, font_size: int, rot: int, cdpt: str, direction: int, char_color = (0,0,0), border_color = (0,255,0), border_size = 2, debug = False) :
	is_pun = _is_punctuation(cdpt)
	cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, direction)
	old_font_size = font_size
	font_size += border_size * 2
	x -= border_size
	y -= border_size
	glyph, empty_char = get_char_glyph(cdpt, old_font_size, direction)
	"""
	glyph_real = glyph.get_glyph()
	stroker = freetype.Stroker()
	stroker.set(64, freetype.FT_STROKER_LINECAP_ROUND, freetype.FT_STROKER_LINEJOIN_ROUND,0)
	#glyph_real.stroke(stroker, True)
	blyph = glyph_real.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0,0), True)
	bitmap = blyph.bitmap
	"""
	offset_x = glyph.advance.x>>6
	offset_y = glyph.advance.y>>6
	bitmap = glyph.bitmap
	
	if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width :
		if offset_y == 0 and direction == 1 :
			offset_y = offset_x
		return offset_x, offset_y
	if debug:
		if offset_y == 0 and direction == 1 :
			offset_y = old_font_size
		return offset_x, offset_y
	print(cdpt, offset_x, offset_y)
	char_map = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
	cv2.imshow("char_map", cv2.resize(char_map, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
	
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
	print(place_x, place_y)
	cv2.imshow("char_map", cv2.resize(char_map, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
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
	
	#cv2.imshow("char_map_alpha", cv2.resize(char_map_alpha, (0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
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

def put_text_horizontal(font_size: int, text: str, lang: str, w: int, h: int, orig_shape: Tuple[int, int], fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	test_image = Image.new("L", (orig_shape[1], orig_shape[0]))
	test_draw = ImageDraw.Draw(test_image)
	if bg :
		if abs(fg[0] - bg[0]) < 40 and abs(fg[1] - bg[1]) < 40 and abs(fg[2] - bg[2]) < 40:
			bg = (255, 255, 255)
	else:
		bg = (255, 255, 255)
	rot = 0
	leading = 1.3
	tracking = 0
	work_parity = 0
	new_font_size = font_size
	while True:
		font = get_font(new_font_size)
		wrapped_text = wrap_text(text, w, font, test_draw) if not lang in ["JPN", "CHS", "CHT"] else "\n".join(list(text))
		spacing = int(new_font_size * (leading - 1))
		stroke_width = int(new_font_size * 0.07)
		x1, y1, x2, y2 = test_draw.multiline_textbbox((0,0), wrapped_text, font=font, spacing=spacing, align="center", stroke_width=stroke_width)
		box_width = x2 - x1
		box_height = y2 - y1
		if box_height > h:
			if work_parity > 0:
				break
			new_font_size -= 2
			work_parity = -1
		elif box_width < w:
			if work_parity < 0:
				break
			new_font_size += 2
			work_parity = 1
		else:
			break
	new_width = math.ceil(box_width)
	new_height = math.ceil(box_height) + stroke_width * 2
	canvas = Image.new("RGBA", (new_width, new_height))
	canvas.putalpha(0)
	draw = ImageDraw.Draw(canvas)
	draw.multiline_text((stroke_width, stroke_width * 2 -spacing), wrapped_text, font=font, spacing=spacing, align="center", stroke_width=stroke_width, fill=fg, stroke_fill=bg)
	#canvas.save("./shill.png")
	numpy_image = np.array(canvas)
	return numpy_image

def prepare_renderer(font_filenames = ['fonts/Arial-Unicode-Regular.ttf', 'fonts/msyh.ttc', 'fonts/msgothic.ttc']) : #'fonts/KoPubWorld Dotum Medium.ttf','fonts/NanumGothic.ttf', 
	global CACHED_FONT_FACE
	for font_filename in font_filenames :
		CACHED_FONT_FACE.append(font_filename)

def test() :
	prepare_renderer()
	put_text_horizontal(64, '안녕, 내 이름은 눈물의 요정! 사회주의 최고의 카레이서다!', "", 360, 830, (2048, 2048), (255, 0, 0), (0, 255, 0))

if __name__ == '__main__' :
	import sys, os 
	p = os.path.abspath('.')
	sys.path.insert(1, p)
	from utils import BBox, Quadrilateral
	test()
