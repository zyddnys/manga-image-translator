import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import List, Union, Tuple

from textblockdetector import TextBlock

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Line:

	def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0) -> None:
		self.text = text
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.length = int(length)
		self.num_words = 0
		if text:
			self.num_words += 1

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
	delta_w = min(x1, im_w - x2, delta_w)
	delta = min(y1, im_h - y2, delta)
	rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
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
		ballon_mask = cv2.erode(ballon_mask, box_kernel, iterations = 1)

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

def seg_eng(text: str) -> List[str]:
	text = text.upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
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

	return processed_text.split(' ')

def layout_lines_with_mask(
	mask: np.ndarray, 
	words: List[str], 
	wl_list: List[int], 
	delimiter_len: int, 
	line_height: int,
	delimiter: str = ' ',
	word_break: bool = False)->List[Line]:

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
		wl_cumsums = np.cumsum(np.array(wl_list, dtype=np.float64))
		wl_cumsums -= wl_cumsums[-1] / 2
		central_index = np.argmin(np.abs(wl_cumsums))
		if wl_list[central_index] < 0:
			central_index += 1
		if central_index > 0:
			wlst_left = words[:central_index]
			len_left = wl_list[:central_index]
			sum_left = np.sum(len_left)
		if central_index < num_words - 1:
			wlst_right = words[central_index + 1:]
			len_right = wl_list[central_index + 1:]
			sum_right = np.sum(len_right)
	else:
		central_index = 0

	pos_y = centroid_y - line_height // 2
	pos_x = centroid_x - wl_list[central_index] // 2

	bh, bw = mask.shape[:2]
	central_line = Line(words[central_index], pos_x, pos_y, wl_list[central_index])
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

	lines = [central_line]

	# layout bottom half
	if sum_right > 0:
		w, wl = wlst_right.pop(0), len_right.pop(0)
		pos_x = centroid_x - wl // 2
		pos_y = centroid_y + line_height // 2
		line_bottom = pos_y + line_height
		line = Line(w, pos_x, pos_y, wl)
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
			else:
				pos_x = centroid_x - wl // 2
				pos_y = line_bottom
				line_bottom += line_height
				line = Line(w, pos_x, pos_y, wl)
				lines.append(line)

	# layout top half
	if sum_left > 0:
		w, wl = wlst_left.pop(-1), len_left.pop(-1)
		pos_x = centroid_x - wl // 2
		pos_y = centroid_y - line_height // 2 - line_height
		line_bottom = pos_y + line_height
		line = Line(w, pos_x, pos_y, wl)
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
			else:
				pos_x = centroid_x - wl // 2
				pos_y -= line_height
				line_bottom = pos_y + line_height
				line = Line(w, pos_x, pos_y, wl)
				lines.insert(0, line)

	return lines

def render_textblock_list_eng(
	img: np.ndarray, 
	blk_list: List[TextBlock], 
	font_path: str, 
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
	
	def get_font(font_size: int):
		fs = int(font_size / (1 + 2*stroke_width))
		font = ImageFont.truetype(font_path, fs)
		sw = int(stroke_width * font.size)
		line_height = font.getmetrics()[0]
		line_height = int(line_height * line_spacing + 2 * sw)
		delimiter_len = int(font.getlength(delimiter))
		base_length = -1
		wl_list = []
		for word in words:
			wl = int(font.getlength(word))
			wl_list.append(wl)
			if wl > base_length:
				base_length = wl
		return font, sw, line_height, delimiter_len, base_length, wl_list
	
	pilimg = Image.fromarray(img)

	for blk in blk_list:
		if blk.vertical:
			blk.angle -= 90
		words = seg_eng(blk.translation)
		num_words = len(words)
		if not num_words:
			continue

		font, sw, line_height, delimiter_len, base_length, wl_list = get_font(blk.font_size)

		if ref_textballon:
			assert original_img is not None
			br = blk.bounding_rect()
			# non-dl textballon segmentation
			enlarge_ratio = min(max(br[2] / br[3], br[3] / br[2]), 3.0)
			ballon_region, ballon_area, xyxy = extract_ballon_region(original_img, br, show_process=False, enlarge_ratio=enlarge_ratio)
			rotated, rx, ry = False, 0, 0
			angle = blk.angle
			if abs(angle) > 3:
				rotated = True
				rad = np.deg2rad(angle)
				r_sin = np.sin(rad)
				r_cos = np.cos(rad)
				rotated_ballon_region = Image.fromarray(ballon_region).rotate(blk.angle, expand=True)
				rotated_ballon_region = np.array(rotated_ballon_region)

				angle %= 360
				if angle > 0 and angle <= 90:
					ry = abs(ballon_region.shape[1] * r_sin)
				elif angle > 90 and angle <= 180:
					rx = abs(ballon_region.shape[1] * r_cos)
					ry = rotated_ballon_region.shape[0]	
				elif angle > 180 and angle <= 270:
					ry = abs(ballon_region.shape[0] * r_cos)
					rx = rotated_ballon_region.shape[1]
				else:
					rx = abs(ballon_region.shape[0] * r_sin)
				ballon_region = rotated_ballon_region

			str_length, _ = font.getsize(blk.translation)
			str_area = str_length * line_height + delimiter_len * (num_words - 1) * line_height
			area_ratio = ballon_area / str_area
			resize_ratio = 1
			if area_ratio < ballonarea_thresh:
				resize_ratio = ballonarea_thresh / area_ratio
				ballon_area = int(resize_ratio * ballon_area)
				resize_ratio = min(np.sqrt(resize_ratio), (1 / downscale_constraint) ** 2)
				rx *= resize_ratio
				ry *= resize_ratio
				ballon_region = cv2.resize(ballon_region, (int(resize_ratio * ballon_region.shape[1]), int(resize_ratio * ballon_region.shape[0])))

			region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(ballon_region))
		
			new_fnt_size = max(region_w / (base_length + 2*sw), downscale_constraint)
			if new_fnt_size < 1:
				font, sw, line_height, delimiter_len, base_length, wl_list = get_font(int(blk.font_size * new_fnt_size))

			lines = layout_lines_with_mask(ballon_region, words, wl_list, delimiter_len, line_height, delimiter)
			
			line_cy = np.array([line.pos_y for line in lines]).mean() + line_height / 2
			region_cy = region_y + region_h / 2
			y_offset = int(round(np.clip(region_cy - line_cy, -line_height, line_height)))

			pos_x_lst, pos_right_lst = [], []
			for line in lines:
				pos_x_lst.append(line.pos_x)
				pos_right_lst.append(max(line.pos_x, 0) + line.length)

			pos_x_lst = np.array(pos_x_lst)
			pos_right_lst = np.array(pos_right_lst)
			canvas_l, canvas_r = pos_x_lst.min() - sw, pos_right_lst.max() + sw
			canvas_t, canvas_b = lines[0].pos_y - sw, lines[-1].pos_y + line_height + sw

			canvas_h = int(canvas_b - canvas_t)
			canvas_w = int(canvas_r - canvas_l)
			lines_map = np.zeros_like(ballon_region, dtype=np.uint8)
			for line in lines:
				# line.pos_y += y_offset
				cv2.rectangle(lines_map, (line.pos_x - sw, line.pos_y + y_offset), (line.pos_x + line.length + sw, line.pos_y + line_height), 255, -1)
				line.pos_x -= canvas_l
				line.pos_y -= canvas_t
			
			raw_lines = render_lines(lines, canvas_h, canvas_w, font, sw, font_color, stroke_color)
			rel_cx = ((canvas_l + canvas_r) / 2 - rx) / resize_ratio
			rel_cy = ((canvas_t + canvas_b) / 2 - ry + y_offset) / resize_ratio

			lines_area = np.sum(lines_map)
			lines_area += (max(0, region_y - canvas_t) + max(0, canvas_b - region_h - region_y)) * canvas_w * 255 \
			 + (max(0, region_x - canvas_l) + max(0, canvas_r - region_w - region_x)) * canvas_h * 255

			valid_lines_ratio = lines_area / np.sum(cv2.bitwise_and(lines_map, ballon_region))
			if valid_lines_ratio > 1:
				resize_ratio = min(resize_ratio * valid_lines_ratio, (1 / downscale_constraint) ** 2)

			if rotated:
				rcx = rel_cx * r_cos - rel_cy * r_sin
				rcy = rel_cx * r_sin + rel_cy * r_cos
				rel_cx = rcx
				rel_cy = rcy
				raw_lines = raw_lines.rotate(-blk.angle, expand=True, resample=Image.BILINEAR)
				raw_lines = raw_lines.crop(raw_lines.getbbox())

			abs_cx = rel_cx + xyxy[0]
			abs_cy = rel_cy + xyxy[1]
			
			if resize_ratio != 1:
				raw_lines = raw_lines.resize((int(raw_lines.width / resize_ratio), int(raw_lines.height / resize_ratio)))
			abs_x = int(abs_cx - raw_lines.width / 2)
			abs_y = int(abs_cy - raw_lines.height / 2)
			pilimg.paste(raw_lines, (abs_x, abs_y), mask=raw_lines)
			# cv2.imshow('ballon_region', ballon_region)
			# cv2.imshow('cropped', original_img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
			# cv2.imshow('raw_lines', np.array(raw_lines))
			# cv2.waitKey(0)
		# older method
		else:
			min_bbox = blk.min_rect(rotate_back=False)[0]
			bx, by = min_bbox[0]
			bw, bh = min_bbox[2] - min_bbox[0]
			cx, cy = bx + bw / 2, by + bh / 2
			base_length = max(base_length, bw)

			pos_x, pos_y = 0, 0
			line = Line(words[0], 0, 0, wl_list[0])
			lines = [line]
			for word, wl in zip(words[1:], wl_list[1:]):
				added_len = int(delimiter_len + wl + line.length)
				if added_len > base_length:
					pos_y += line_height
					line = Line(word, 0, pos_y, wl)
					lines.append(line)
				else:
					line.text = line.text + ' ' + word
					line.length = added_len
			last_line = lines[-1]
			canvas_h = last_line.pos_y + line_height
			canvas_w = int(base_length)

			for line in lines:
				line.pos_x = int((base_length - line.length) / 2) if align_center else 0
			raw_lines = render_lines(lines, canvas_h, canvas_w, font, sw, font_color, stroke_color)

			if abs(blk.angle) > 3:
				raw_lines = raw_lines.rotate(-blk.angle, expand=True)
			im_w, im_h = raw_lines.size
			scale = max(min(bh / im_h * size_tol, bw / im_w * size_tol), downscale_constraint)
			if scale < 1:
				raw_lines = raw_lines.resize((int(im_w*scale), int(im_h*scale)))

			im_w, im_h = raw_lines.size
			paste_x, paste_y = int(cx - im_w / 2), int(cy - im_h / 2)
			
			pilimg.paste(raw_lines, (paste_x, paste_y), mask=raw_lines)

	# pilimg.show()
	return np.array(pilimg)