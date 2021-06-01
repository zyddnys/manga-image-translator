
from itertools import filterfalse
import pickle
from typing import List, Tuple, Optional

import numpy as np

import cv2
import unicodedata
import freetype

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

def add_color(bw_char_map, color, border_color = None, border_size: int = 0, bg = None) :
	char_intensity = np.tile(bw_char_map[:,:,None], 3).astype(np.float32) / 255.0
	bg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 3), dtype = np.uint8) if bg is None else bg
	fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 3), dtype = np.uint8)
	bg_mask = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1]), dtype = np.uint8)
	if border_size > 0 :
		bg_mask = np.copy(bw_char_map)
		kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_size + 1, 2 * border_size + 1))
		bg_mask = cv2.dilate(bg_mask, kern)
	color_np = np.array(color, dtype = np.uint8)
	if border_color and border_size > 0 :
		mask = bg_mask.astype(bool) | (bw_char_map > 0)
		bg_color_np = np.array(border_color, dtype = np.uint8)
		bg[mask] = bg_color_np
	else :
		mask = bw_char_map > 0
	fg[mask] = color_np
	return (bg * (1 - char_intensity) + fg * char_intensity).astype(np.uint8), mask, color, border_color if border_size > 0 else None

CACHED_FONT_FACE = None

def get_char_glyph(cdpt, font_size: int, direction: int) :
	global CACHED_FONT_FACE
	face = CACHED_FONT_FACE

	if direction == 0 :
		face.set_pixel_sizes( 0, font_size )
	elif direction == 1 :
		face.set_pixel_sizes( font_size, 0 )
	face.load_char(cdpt)
	return face.glyph, face.glyph.bitmap.rows * face.glyph.bitmap.width == 0

def put_char(canvas: np.ndarray, x: int, y: int, font_size: int, rot: int, cdpt: str, direction: int, char_color = (0,0,0), border_color = (0,255,0), border_size = 2) :
	is_pun = _is_punctuation(cdpt)
	cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, direction)
	glyph, empty_char = get_char_glyph(cdpt, font_size, direction)
	offset_x = glyph.advance.x>>6
	offset_y = glyph.advance.y>>6
	bitmap = glyph.bitmap
	if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width :
		if offset_y == 0 and direction == 1 :
			offset_y = offset_x
		return (offset_x, offset_y), (0, 0, 0, 0), True, 0, char_color, border_color

	char_map = np.array(bitmap.buffer, dtype = np.uint8).reshape((bitmap.rows,bitmap.width))
	size = font_size
	if is_pun :
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
	degree = 0
	if rot == 1 and direction == 1 :
		degree = -90.0
	elif rot == 2 :
		degree = 0
	if rot_degree != 0 :
		degree = rot_degree
	char_map, (diff_i, diff_j) = rotate_image(char_map, degree)
	char_map = char_map[(diff_i - y if diff_i > y else 0):, (diff_j - x if diff_j > x else 0):]
	diff_i = y if diff_i > y else diff_i
	diff_j = x if diff_j > x else diff_j
	available_shape = canvas[y - diff_i:y+diff_i+font_size,x-diff_j: x+diff_j+font_size,:].shape
	char_map = char_map[:available_shape[0],:available_shape[1]]
	#char_map_mask = np.tile((char_map > 0)[:,:,None], 3)
	if len(char_map.shape) == 3 :
		char_map = char_map.squeeze(-1)
	if border_color :
		char_map, char_map_mask, char_color, border_color = add_color(char_map, char_color, border_color=border_color, border_size=border_size)
	else :
		char_map, char_map_mask, char_color, border_color = add_color(char_map, char_color, bg = canvas[y - diff_i:y+diff_i+font_size,x-diff_j: x+diff_j+font_size,:])
	x2, y2, w2, h2 = cv2.boundingRect(char_map_mask.astype(np.uint8) * 255)
	#if w2 * h2 == 0 :
	#	print(f'[*] "{cdpt}" is Empty character')
	#char_map = cv2.resize(char_map, (font_size, font_size), interpolation=cv2.INTER_LANCZOS4)
	np.putmask(canvas[y - diff_i:y+diff_i+font_size,x-diff_j: x+diff_j+font_size,:], np.tile(char_map_mask[:,:,None], 3), char_map)
	#cv2.rectangle(canvas, (x-diff_j+x2, y - diff_i+y2), (x-diff_j+x2+w2, y - diff_i+y2+h2), (255,0,0))
	#canvas[y - diff_i:y+diff_i+font_size, x-diff_j: x+diff_j+font_size, :] = np.tile(char_map_mask[:,:,None], 3) * char_map
	#canvas[y:y+font_size,x: x+font_size,:]=255-char_map[:font_size,:font_size]#np.array([0,0,0],dtype=np.uint8)
	# if offset_x == 0 and direction == 0 :
	# 	offset_x = max(char_map.shape[1], offset_y)
	if offset_y == 0 and direction == 1 :
		#print(f'0 offset_y for {cdpt}')
		offset_y = font_size#h2#min(h2+y2, font_size)
		# if my_randint(0,1) == 0 :
		# 	offset_y = min(h2, font_size) #bitmap.rows+place_y
		# else :
		# 	offset_y = offset_x#bitmap.rows+place_y*2#max(char_map.shape[0], offset_x)
	return (offset_x, offset_y), (x-diff_j+x2, y - diff_i+y2, w2, h2), False, degree, char_color, border_color

def put_text_vertical(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	x1 = x
	x2 = x + w
	y1 = y
	y2 = y + h
	#cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	font_size = round(w / (line_count * 1.1))
	rows = h // font_size
	cols = w // font_size
	while rows * cols < len(text) :
		font_size -= 1
		rows = h // font_size
		cols = w // font_size
	bgsize = int(max(font_size * 0.025, 1)) if bg else 0
	spacing_y = int(max(font_size * 0.05, 1))
	spacing_x = spacing_y
	x = x2 - spacing_x - font_size
	y = y1 + max(spacing_y, 0)
	txt_i = 0
	rot = 0
	for j in range(cols) :
		new_length = rows
		if not new_length :
			continue
		y = y1 + spacing_y
		for i in range(new_length) :
			(x_offset, y_offset), (ch_x, ch_y, ch_w, ch_h), empty_ch, degree, char_color, border_color = put_char(img, x, y, font_size, rot, text[txt_i], 1, char_color=fg,border_color=bg,border_size=bgsize)
			fg = char_color
			txt_i += 1
			if txt_i >= len(text) :
				return True
			y += spacing_y + y_offset
		x -= spacing_x + font_size
	return True

def put_text_horizontal(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	x1 = x
	x2 = x + w
	y1 = y
	y2 = y + h
	#cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	font_size = round(h / (line_count * 1.1))
	rows = h // font_size
	cols = w // font_size
	while rows * cols < len(text) :
		font_size -= 1
		rows = h // font_size
		cols = w // font_size
	bgsize = int(max(font_size * 0.025, 1)) if bg else 0
	spacing_x = int(max(font_size * 0.05, 1))
	spacing_y = spacing_x
	x = x1 + max(spacing_x, 0)
	y = y1 + spacing_y
	txt_i = 0
	rot = 0
	for i in range(rows) :
		new_length = cols
		if not new_length :
			continue
		x = x1 + spacing_x
		for j in range(new_length) :
			(x_offset, y_offset), (ch_x, ch_y, ch_w, ch_h), empty_ch, degree, char_color, border_color = put_char(img, x, y, font_size, rot, text[txt_i], 0, char_color=fg,border_color=bg,border_size=bgsize)
			fg = char_color
			txt_i += 1
			if txt_i >= len(text) :
				return True
			x += spacing_x + x_offset
		y += font_size + spacing_y
	return True

def put_text(img: np.ndarray, text: str, line_count: int, x: int, y: int, w: int, h: int, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]]) :
	pass

def prepare_renderer(font_filename = 'Arial-Unicode-Regular.ttf') :
	global CACHED_FONT_FACE
	CACHED_FONT_FACE = freetype.Face(font_filename)
	pass

def test() :
	prepare_renderer()
	canvas = np.ones((4096, 2590, 3), dtype = np.uint8) * 255
	put_text_vertical(canvas, '因为不同‼ 这真的是普通的肉！那个姑娘的恶作剧！是吗？咲夜⁉', 4, 2143, 3219, 355, 830, (0, 0, 0), None)
	put_text_horizontal(canvas, '添加幽默FOR foy', 1, 242, 87, 2093, 221, (0, 0, 0), None)
	cv2.imwrite('text_render_combined.png', canvas)

def find_ja_fonts() :
	global CDPT_FONT_MAP
	with open('char_font_map_2339fonts_v5.pkl', 'rb') as fp :
		CDPT_FONT_MAP = pickle.load(fp)
	s = set(CDPT_FONT_MAP['…'])
	print(s)
	for ch in ['…', '？', '！', '♥', '~'] :
		s = s.intersection(set(CDPT_FONT_MAP[ch]))
		if not s :
			print(ch)
		#print(s)
	print(s)

if __name__ == '__main__' :
	find_ja_fonts()
