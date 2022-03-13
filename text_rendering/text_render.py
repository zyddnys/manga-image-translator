if __name__ == '__main__' :
	import sys, os
	sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from itertools import filterfalse
import pickle
from typing import List, Tuple, Optional

import numpy as np
import cv2
import unicodedata
import freetype
from utils import BBox, Quadrilateral
from PIL import Image, ImageFont, ImageDraw
import math
import textwrap

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
			return cdpt, 0
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

def add_color(bw_char_map, color, stroke_char_map, stroke_color) :
	fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype = np.uint8)
	if bw_char_map.size == 0 :
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
		self.metrics = namespace()
		self.metrics.vertBearingX = glyph.metrics.vertBearingX
		self.metrics.vertBearingY = glyph.metrics.vertBearingY
		self.metrics.horiAdvance = glyph.metrics.horiAdvance
		self.metrics.vertAdvance = glyph.metrics.vertAdvance

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
		return Glyph(face.glyph)

def get_char_glyph_orig(cdpt, font_size: int, direction: int) :
	global CACHED_FONT_FACE
	for i, face in enumerate(CACHED_FONT_FACE) :
		if face.get_char_index(cdpt) == 0 and i != len(CACHED_FONT_FACE) - 1 :
			continue
		if direction == 0 :
			face.set_pixel_sizes( 0, font_size )
		elif direction == 1 :
			face.set_pixel_sizes( font_size, 0 )
		face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
		return face.glyph

def get_font(font_size: int, direction=0) :
	font_filenames = ['fonts/Arial-Unicode-Regular.ttf', 'fonts/msyh.ttc', 'fonts/msgothic.ttc']
	for face in font_filenames :
		return ImageFont.truetype(face, font_size)

def calc_char_vertical(font_size: int, rot: int, text: str, max_height: int, border_size = 2) :
	line_text_list = []
	line_max_width_list = []
	line_center_list = []
	line_height_list = []
	line_char_info_list = []
	line_height = 0
	line_str = ""
	line_width_left = 0
	line_width_right = 0
	for i, cdpt in enumerate(text):
		is_pun = _is_punctuation(cdpt)
		cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
		glyph = get_char_glyph(cdpt, font_size, 1)
		#offset_x = glyph.advance.x>>6
		#offset_y = glyph.advance.y>>6
		bitmap = glyph.bitmap
		# spaces, etc
		if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width :
			char_offset_y = glyph.metrics.vertBearingY >> 6
		else :
			char_offset_y = (glyph.metrics.vertAdvance >> 6) + border_size * 2
		char_width = bitmap.width + border_size * 2
		#char_height = bitmap.rows + border_size * 2
		char_bearing_x = glyph.metrics.vertBearingX >> 6
		#char_bearing_y = glyph.metrics.vertBearingY
		if line_height + char_offset_y > max_height:
			line_text_list.append(line_str)
			line_height_list.append(line_height)
			line_max_width_list.append(line_width_left + line_width_right)
			line_center_list.append(line_width_left)
			line_str = ""
			line_height = 0
			line_width_left = 0
			line_width_left = 0
		line_height += char_offset_y
		line_str += cdpt
		line_width_left = max(line_width_left, abs(char_bearing_x) + border_size)
		line_width_right = max(line_width_right, char_width + border_size - abs(char_bearing_x))
	# last char
	line_text_list.append(line_str)
	line_height_list.append(line_height)
	line_max_width_list.append(line_width_left + line_width_right)
	line_center_list.append(line_width_left)
	return line_text_list, line_max_width_list, line_height_list, line_center_list

def put_char_vertical(font_size: int, rot: int, line_text: str, line_width: int, line_height: int, line_center: int, char_color = (0,0,0), border_color = (0,255,0), border_size = 2) :
	line_box = np.zeros((line_height, line_width), dtype=np.uint8)
	line_border_box = line_box.copy()
	y = 0
	for cdpt in line_text:
		is_pun = _is_punctuation(cdpt)
		cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
		glyph = get_char_glyph(cdpt, font_size, 1)
		#offset_x = glyph.advance.x>>6
		#offset_y = glyph.advance.y>>6
		bitmap = glyph.bitmap
		char_map = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
		if border_color and border_size > 0:
			slot = get_char_glyph_orig(cdpt, font_size, 1)
			border_glyph = slot.get_glyph()
			stroker = freetype.Stroker()
			stroker.set(64*border_size, freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINEJOIN_ROUND, 0)
			border_glyph.stroke(stroker, destroy=True)
			blyph = border_glyph.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0,0), True)
			border_bitmap = blyph.bitmap
			if (border_bitmap.rows * border_bitmap.width == 0 or len(border_bitmap.buffer) != border_bitmap.rows * border_bitmap.width) :
				y += slot.metrics.vertBearingY >> 6
				continue
			place_x = line_center + (slot.metrics.vertBearingX >> 6)
			place_y = y + (slot.metrics.vertBearingY >> 6)
			line_border_box[place_y:place_y+border_bitmap.rows, place_x:place_x+border_bitmap.width] = np.array(border_bitmap.buffer, dtype = np.uint8).reshape(border_bitmap.rows,border_bitmap.width)
		place_x += border_size
		place_y += border_size
		line_box[place_y:place_y+bitmap.rows, place_x:place_x+bitmap.width] = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
		y += (slot.metrics.vertAdvance >> 6) + border_size * 2
	if border_color and border_size > 0:
		line_box = add_color(line_box, char_color, line_border_box, border_color)
	else:
		line_box = add_color(line_box, char_color)
	return line_box

def put_text_vertical(font_size: int, mag_ratio: float, text: str, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	bgsize = int(max(font_size * 0.07, 1)) if bg else 0
	spacing_y = 0
	spacing_x = int(max(font_size * 0.2, 0))
	rot = 0

	# pre-calculate line breaks
	line_text_list, line_width_list, line_height_list, line_center_list = calc_char_vertical(font_size, rot, text, h, border_size=bgsize)
	# make box
	box = np.zeros((max(line_height_list), sum(line_width_list) + (len(line_width_list) - 1) * spacing_x, 4),dtype=np.uint8)
	x = box.shape[1]
	# put text
	for j, (line_text, line_width, line_height, line_center) in enumerate(zip(line_text_list, line_width_list, line_height_list, line_center_list)):
		line_bitmap = put_char_vertical(font_size, rot, line_text, line_width, line_height, line_center, char_color=fg,border_color=bg,border_size=bgsize)
		x -= line_bitmap.shape[1]
		box[0:line_bitmap.shape[0],x:x+line_bitmap.shape[1]] = line_bitmap
		x -= spacing_x
	return box

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

def put_text_horizontal(font_size: int, mag_ratio: float, text: str, w: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	bgsize = int(max(font_size * 0.07, 1)) if bg else 0
	spacing_y = int(max(font_size * 0.2, 0))
	#spacing_x = 0
	rot = 0

	leading = 1.2
	#tracking = 0
	font = get_font(font_size)

	test_image = Image.new("L", (10000,10000))
	test_draw = ImageDraw.Draw(test_image)
	wrapped_text = "\n".join(textwrap.wrap(text, width=w // font_size))#wrap_text(text, w, font, test_draw)
	x1, y1, x2, y2 = test_draw.multiline_textbbox((0,0), wrapped_text, font=font, spacing=spacing_y, align="center", stroke_width=bgsize)
	box_width = x2 - x1
	box_height = y2 - y1
	new_width = math.ceil(box_width)
	new_height = math.ceil(box_height) + bgsize * 2
	canvas = Image.new("RGBA", (new_width, new_height))
	canvas.putalpha(0)
	draw = ImageDraw.Draw(canvas)
	draw.multiline_text((bgsize, bgsize * 2 - spacing_y), wrapped_text, font=font, spacing=spacing_y, align="center", stroke_width=bgsize, fill=fg, stroke_fill=bg)
	#canvas.save("./shill.png")
	numpy_image = np.array(canvas)
	return numpy_image

def put_text(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	pass

def prepare_renderer(font_filenames = ['fonts/Arial-Unicode-Regular.ttf', 'fonts/msyh.ttc', 'fonts/msgothic.ttc']) :
	global CACHED_FONT_FACE
	for font_filename in font_filenames :
		CACHED_FONT_FACE.append(freetype.Face(font_filename))

def test() :
	prepare_renderer()
	#font_size: int, mag_ratio: float, img: np.ndarray, mask: np.ndarray, text: str, line_count: int, lines: List[Quadrilateral], x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]
	#canvas = np.ones((4096, 2590, 4), dtype = np.uint8) * 255
	#mask = np.zeros((4096, 2590), dtype = np.uint8)
	#《因为不同‼ [这"真的是普]通的》肉！那个“姑娘”的恶作剧！是吗？咲夜⁉
	canvas = put_text_vertical(64, 1.0, '因为不同‼ [这"真的是普]通的》肉！那个“姑娘”的恶作剧！是吗？咲夜⁉', 1000, (0, 0, 0), (255, 128, 128))
	

	#put_text_horizontal(canvas, '“添加幽默”FOR if else !?xxj', 1, [], 242, 87, 2093, 221, (0, 0, 0), None)
	cv2.imwrite('text_render_combined.png', canvas)

if __name__ == '__main__' :
	test()
