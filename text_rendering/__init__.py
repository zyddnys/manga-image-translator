
from typing import List
from utils import Quadrilateral
import numpy as np
import cv2
from utils import findNextPowerOf2

from . import text_render
from textblockdetector.textblock import TextBlock

async def dispatch(img_canvas: np.ndarray, text_mag_ratio: np.integer, translated_sentences: List[str], textlines: List[Quadrilateral], text_regions: List[Quadrilateral], text_direction_overwrite: str) -> np.ndarray :
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		if not trans_text :
			continue
		if text_direction_overwrite and text_direction_overwrite in ['h', 'v'] :
			region.majority_dir = text_direction_overwrite
		print(region.text)
		print(trans_text)
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
		#img_bbox = cv2.polylines(img_bbox, [region.pts], True, color=(0, 0, 255), thickness = 2)

		region_aabb = region.aabb
		print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)

		# round font_size to fixed powers of 2, so later LRU cache can work
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

		tmp_canvas = np.ones((enlarged_h * 2, enlarged_w * 2, 3), dtype = np.uint8) * 127
		tmp_mask = np.zeros((enlarged_h * 2, enlarged_w * 2), dtype = np.uint16)

		if region.majority_dir == 'h' :
			text_render.put_text_horizontal(
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

		tmp_mask = np.clip(tmp_mask, 0, 255).astype(np.uint8)
		x, y, w, h = cv2.boundingRect(tmp_mask)
		r_prime = w / h
		r = region.aspect_ratio
		w_ext = 0
		h_ext = 0
		if r_prime > r :
			h_ext = w / (2 * r) - h / 2
		else :
			w_ext = (h * r - w) / 2
		region_ext = round(min(w, h) * 0.05)
		h_ext += region_ext
		w_ext += region_ext
		src_pts = np.array([[x - w_ext, y - h_ext], [x + w + w_ext, y - h_ext], [x + w + w_ext, y + h + h_ext], [x - w_ext, y + h + h_ext]]).astype(np.float32)
		src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
		src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)
		dst_pts = region.pts
		M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		tmp_rgba = np.concatenate([tmp_canvas, tmp_mask[:, :, None]], axis = -1).astype(np.float32)
		rgba_region = np.clip(cv2.warpPerspective(tmp_rgba, M, (img_canvas.shape[1], img_canvas.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
		canvas_region = rgba_region[:, :, 0: 3]
		mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
		img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
	return img_canvas


async def dispatch_ctd_render(img_canvas: np.ndarray, text_mag_ratio: np.integer, translated_sentences: List[str], text_regions: List[TextBlock], text_direction_overwrite: str) -> np.ndarray :
	for ridx, (trans_text, region) in enumerate(zip(translated_sentences, text_regions)) :
		print(f'text: {region.get_text()} \n trans: {trans_text}')
		if not trans_text :
			continue
		if text_direction_overwrite and text_direction_overwrite in ['h', 'v'] :
			majority_dir = text_direction_overwrite
		else:
			majority_dir = 'v' if region.vertical else 'h'

		fg, bg = region.get_font_colors()
		font_size = region.font_size
		font_size = round(font_size)

		region_x, region_y, region_w, region_h = region.xyxy
		region_w -= region_x
		region_h -= region_y

		textlines = []
		for ii, text in enumerate(region.text):
			textlines.append(Quadrilateral(np.array(region.lines[ii]), text, 1, region.fg_r, region.fg_g, region.fg_b, region.bg_r, region.bg_g, region.bg_b))
		# region_aabb = region.aabb
		# print(region_aabb.x, region_aabb.y, region_aabb.w, region_aabb.h)

		# round font_size to fixed powers of 2, so later LRU cache can work
		font_size_enlarged = findNextPowerOf2(font_size) * text_mag_ratio
		enlarge_ratio = font_size_enlarged / font_size
		font_size = font_size_enlarged
		while True :
			enlarged_w = round(enlarge_ratio * region_w)
			enlarged_h = round(enlarge_ratio * region_h)
			rows = enlarged_h // (font_size * 1.3)
			cols = enlarged_w // (font_size * 1.3)
			if rows * cols < len(trans_text) :
				enlarge_ratio *= 1.1
				continue
			break
		print('font_size:', font_size)

		tmp_canvas = np.ones((enlarged_h * 2, enlarged_w * 2, 3), dtype = np.uint8) * 127
		tmp_mask = np.zeros((enlarged_h * 2, enlarged_w * 2), dtype = np.uint16)

		if majority_dir == 'h' :
			text_render.put_text_horizontal(
				font_size,
				enlarge_ratio * 1.0,
				tmp_canvas,
				tmp_mask,
				trans_text,
				len(region.lines),
				textlines,
				enlarged_w // 2,
				enlarged_h // 2,
				enlarged_w,
				enlarged_h,
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
				len(region.lines),
				textlines,
				enlarged_w // 2,
				enlarged_h // 2,
				enlarged_w,
				enlarged_h,
				fg,
				bg
			)

		tmp_mask = np.clip(tmp_mask, 0, 255).astype(np.uint8)
		x, y, w, h = cv2.boundingRect(tmp_mask)
		r_prime = w / h

		r = region.aspect_ratio()
		if majority_dir != 'v':
			r = 1 / r

		w_ext = 0
		h_ext = 0
		if r_prime > r :
			h_ext = w / (2 * r) - h / 2
		else :
			w_ext = (h * r - w) / 2
		region_ext = round(min(w, h) * 0.05)
		h_ext += region_ext
		w_ext += region_ext
		src_pts = np.array([[x - w_ext, y - h_ext], [x + w + w_ext, y - h_ext], [x + w + w_ext, y + h + h_ext], [x - w_ext, y + h + h_ext]]).astype(np.float32)
		src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
		src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)
		
		dst_pts = region.min_rect()
		if majority_dir == 'v':
			dst_pts = dst_pts[:, [3, 0, 1, 2]]
		M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		tmp_rgba = np.concatenate([tmp_canvas, tmp_mask[:, :, None]], axis = -1).astype(np.float32)
		rgba_region = np.clip(cv2.warpPerspective(tmp_rgba, M, (img_canvas.shape[1], img_canvas.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0), 0, 255)
		canvas_region = rgba_region[:, :, 0: 3]
		mask_region = rgba_region[:, :, 3: 4].astype(np.float32) / 255.0
		img_canvas = np.clip((img_canvas.astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
	return img_canvas