
from typing import List
from utils import Quadrilateral
import numpy as np
import cv2
from utils import findNextPowerOf2

from . import text_render

async def dispatch(img_canvas: np.ndarray, text_mag_ratio: np.integer, translated_sentences: List[str], textlines: List[Quadrilateral], text_regions: List[Quadrilateral]) -> np.ndarray :
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		if not trans_text :
			continue
		#print(region.majority_dir, region.pts)
		fg = (region.fg_r, region.fg_g, region.fg_b)
		bg = (region.bg_r, region.bg_g, region.bg_b)


		font_size = 0
		n_lines = len(region.textline_indices)
		for idx in region.textline_indices :
			txtln = textlines[idx]
			#img_bbox = cv2.polylines(img_bbox, [txtln.pts], True, color = fg, thickness=2)
			# [l1a, l1b, l2a, l2b] = txtln.structure
			# cv2.line(img_bbox, l1a, l1b, color = (0, 255, 0), thickness = 2)
			# cv2.line(img_bbox, l2a, l2b, color = (0, 0, 255), thickness = 2)
			#dbox = txtln.aabb
			font_size = max(font_size, txtln.font_size)
			#cv2.rectangle(img_bbox, (dbox.x, dbox.y), (dbox.x + dbox.w, dbox.y + dbox.h), color = (255, 0, 255), thickness = 2)
		font_size = round(font_size)
		font_size = font_size if font_size % 2 == 0 else font_size + 1
		#img_bbox = cv2.polylines(img_bbox, [region.pts], True, color=(0, 0, 255), thickness = 2)

		region_aabb = region.aabb
		
		print(region.text)
		print(trans_text)
		print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)
		# round font_size to fixed powers of 2, so later LRU cache can work
		"""
		font_size_enlarged = findNextPowerOf2(font_size) * text_mag_ratio
		enlarge_ratio = font_size_enlarged / font_size
		font_size = font_size_enlarged
		while True :
			enlarged_w = round(enlarge_ratio * region_aabb.w)
			enlarged_h = round(enlarge_ratio * region_aabb.h)
			rows = enlarged_h // (font_size * 1.3)
			cols = enlarged_w // (font_size * 1.3)
			if rows * cols < len(trans_text) :
				enlarge_ratio *= 1.1
				continue
			break
		print('font_size:', font_size)
		print('enlarge_ratio:', enlarge_ratio)
		"""
		#font_size = findNextPowerOf2(font_size)
		enlarged_w = region_aabb.w
		enlarged_h = region_aabb.h
		enlarge_ratio = 1
		#tmp_canvas = np.ones((enlarged_h * 2, enlarged_w * 2, 3), dtype = np.uint8) * 127
		#tmp_mask = np.zeros((enlarged_h * 2, enlarged_w * 2), dtype = np.uint16)
		#tmp_canvas = np.ones((img_canvas.shape[1], img_canvas.shape[0], 3), dtype = np.uint8) * 127
		#tmp_mask = np.zeros((img_canvas.shape[1], img_canvas.shape[0]), dtype = np.uint16)

		if region.majority_dir == 'h' or True :
			tmp_rgba = text_render.put_text_horizontal(
				font_size,
				trans_text,
				"",
				region_aabb.w,
				region_aabb.h,
				img_canvas.shape[:-1],
				fg,
				bg
			)
		else :
			text_render.put_text_vertical(
				font_size,
				enlarge_ratio * 1.0,
				tmp_canvas,
				tmp_mask,
				trans_text,
				len(region.textline_indices),
				[textlines[idx] for idx in region.textline_indices],
				enlarged_w // 2,
				enlarged_h // 2,
				enlarged_w,
				enlarged_h,
				fg,
				bg
			)
		x = 0
		y = 0
		w = tmp_rgba.shape[1]
		h = tmp_rgba.shape[0]
		target_x = region_aabb.x - int((w - region_aabb.w)/2)
		target_y = region_aabb.y + int((region_aabb.h - h)/2)
		target_x = max(min(target_x + tmp_rgba.shape[1], img_canvas.shape[1]) - tmp_rgba.shape[1] , 0)
		target_y = max(min(target_y + tmp_rgba.shape[0], img_canvas.shape[0]) - tmp_rgba.shape[0] , 0)
		rgba_region = np.zeros((img_canvas.shape[0], img_canvas.shape[1],4), np.uint8)
		rgba_region[target_y:target_y+ tmp_rgba.shape[0], target_x:target_x+ tmp_rgba.shape[1]] = tmp_rgba
		canvas_region = rgba_region[:, :, 0: 3]
		mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
		img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
	return img_canvas
