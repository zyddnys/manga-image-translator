import math
import os
import json
import sys
import time
import cv2 as cv
import numpy as np

from .panel import Panel
from .segment import Segment
from .debug import Debug


class NotAnImageException(Exception):
	pass


class Page:

	DEFAULT_MIN_PANEL_SIZE_RATIO = 1 / 10

	def get_infos(self):
		actual_gutters = self.actual_gutters()

		return {
			'filename': self.url if self.url else os.path.basename(self.filename),
			'size': self.img_size,
			'numbering': self.numbering,
			'gutters': [actual_gutters['x'], actual_gutters['y']],
			'license': self.license,
			'panels': list(map(lambda p: p.to_xywh(), self.panels)),
			'processing_time': self.processing_time
		}

	def __init__(
		self,
		filename,
		numbering = None,
		debug = False,
		url = None,
		min_panel_size_ratio = None,
		panel_expansion = True
	):
		self.filename = filename
		self.panels = []
		self.segments = []

		self.processing_time = None
		t1 = time.time_ns()

		self.img = cv.imread(filename)
		if not isinstance(self.img, np.ndarray) or self.img.size == 0:
			raise NotAnImageException(f"File {filename} is not an image")

		self.numbering = numbering or "ltr"
		if not (numbering in ['ltr', 'rtl']):
			raise Exception('Fatal error, unknown numbering: ' + str(numbering))

		self.small_panel_ratio = min_panel_size_ratio or Page.DEFAULT_MIN_PANEL_SIZE_RATIO
		self.panel_expansion = panel_expansion
		self.url = url

		self.img_size = list(self.img.shape[:2])
		self.img_size.reverse()  # get a [width,height] list

		Debug.contour_size = 3

		# get license for this file
		self.license = None
		if os.path.isfile(filename + '.license'):
			with open(filename + '.license', encoding = "utf8") as fh:
				try:
					self.license = json.load(fh)
				except json.decoder.JSONDecodeError:
					print(f"License file {filename+'.license'} is not a valid JSON file", file = sys.stderr)
					sys.exit(1)

		Debug.set_base_img(self.img)

		Debug.add_step('Initial state', self.get_infos())
		Debug.add_image('Input image')

		self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
		Debug.add_image('Shades of gray', img = self.gray)
		Debug.show_time("Shades of gray")

		# https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
		ddepth = cv.CV_16S
		grad_x = cv.Sobel(self.gray, ddepth, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)
		# Gradient-Y
		# grad_y = cv.Scharr(self.gray,ddepth,0,1)
		grad_y = cv.Sobel(self.gray, ddepth, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)

		abs_grad_x = cv.convertScaleAbs(grad_x)
		abs_grad_y = cv.convertScaleAbs(grad_y)

		self.sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
		Debug.add_image('Sobel filter applied', img = self.sobel)
		Debug.show_time("Sobel filter")

		self.get_contours()
		self.get_segments()
		self.get_initial_panels()
		self.group_small_panels()
		self.split_panels()
		self.exclude_small_panels()
		self.merge_panels()
		self.deoverlap_panels()
		self.exclude_small_panels()

		if self.panel_expansion:
			self.panels.sort()  # TODO: move this below before panels sort-fix, when panels expansion is smarter
			self.expand_panels()

		if len(self.panels) == 0:
			self.panels.append(Panel(page = self, xywh = [0, 0, self.img_size[0], self.img_size[1]]))

		self.group_big_panels()

		self.fix_panels_numbering()

		self.processing_time = int((time.time_ns() - t1) / 10**7) / 100

	def get_contours(self):
		# Black background: values above 100 will be black, the rest white
		_, thresh = cv.threshold(self.sobel, 100, 255, cv.THRESH_BINARY)
		Debug.show_time("Image threshhold")

		self.contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]

		Debug.add_image("Thresholded image", img = thresh)
		Debug.show_time("Get contours")

	def get_segments(self):
		self.segments = None

		lsd = cv.createLineSegmentDetector(0)
		dlines = lsd.detect(self.gray)

		Debug.show_time("Detected segments")

		min_dist = min(self.img_size) * self.small_panel_ratio

		while self.segments is None or len(self.segments) > 500:
			self.segments = []

			if dlines is None or dlines[0] is None:
				break

			for dline in dlines[0]:
				x0 = int(round(dline[0][0]))
				y0 = int(round(dline[0][1]))
				x1 = int(round(dline[0][2]))
				y1 = int(round(dline[0][3]))

				a = x0 - x1
				b = y0 - y1
				dist = math.sqrt(a**2 + b**2)
				if dist >= min_dist:
					self.segments.append(Segment([x0, y0], [x1, y1]))

			min_dist *= 1.1

		self.segments = Segment.union_all(self.segments)

		Debug.draw_segments(self.segments, Debug.colours['green'])
		Debug.add_image("Segment Detector")
		Debug.show_time("Compiled segments")

	# Get (square) panels out of initial contours
	def get_initial_panels(self):
		self.panels = []
		for contour in self.contours:
			arclength = cv.arcLength(contour, True)
			epsilon = 0.001 * arclength
			approx = cv.approxPolyDP(contour, epsilon, True)

			panel = Panel(page = self, polygon = approx)
			if panel.is_very_small():
				continue

			Debug.draw_contours([approx], Debug.colours['red'])

			self.panels.append(panel)

		Debug.add_image('Initial contours')
		Debug.add_step('Panels from initial contours', self.get_infos())

	# Group small panels that are close together, into bigger ones
	def group_small_panels(self):
		small_panels = list(filter(lambda p: p.is_small(), self.panels))
		groups = {}
		group_id = 0

		for i, p1 in enumerate(small_panels):
			for p2 in small_panels[i + 1:]:
				if p1 == p2:
					continue

				if not p1.is_close(p2):
					continue

				if p1 not in groups and p2 not in groups:
					group_id += 1
					groups[p1] = group_id
					groups[p2] = group_id
				elif p1 in groups and p2 not in groups:
					groups[p2] = groups[p1]
				elif p2 in groups and p1 not in groups:
					groups[p1] = groups[p2]
				elif groups[p1] != groups[p2]:
					# group group1 and group2 together
					for p, id in groups.items():
						if id == groups[p2]:
							groups[p] = groups[p1]

		grouped = {}
		for k, v in groups.items():
			grouped[v] = grouped.get(v, []) + [k]

		for small_panels in grouped.values():
			big_hull = cv.convexHull(np.concatenate(list(map(lambda p: p.polygon, small_panels))))
			big_panel = Panel(page = self, polygon = big_hull, splittable = False)

			self.panels.append(big_panel)
			for p in small_panels:
				self.panels.remove(p)

			Debug.draw_contours(list(map(lambda p: p.polygon, small_panels)), Debug.colours['lightblue'])
			Debug.draw_contours([big_panel.polygon], Debug.colours['red'])

		if group_id > 0:
			Debug.add_image('Group small panels')
		Debug.add_step('Group small panels', self.get_infos())

	# See if panels can be cut into several (two non-consecutive points are close)
	def split_panels(self):
		did_split = True
		while did_split:
			did_split = False
			for p in sorted(self.panels, key = lambda p: p.area(), reverse = True):
				split = p.split()
				if split is not None:
					did_split = True
					self.panels.remove(p)
					self.panels += split.subpanels

					Debug.draw_contours(list(map(lambda n: n.polygon, split.subpanels)), Debug.colours['blue'])
					Debug.draw_line(split.segment.a, split.segment.b, Debug.colours['red'])
					break

			if did_split:
				Debug.add_image(
					'Split contours (blue contours, red split-segment, gray polygon dots, purple nearby dots)'
				)

		Debug.add_step(f"Panels from split contours ({len(self.segments)} segments)", self.get_infos())

	def exclude_small_panels(self):
		self.panels = list(filter(lambda p: not p.is_small(), self.panels))

		Debug.add_step('Exclude small panels', self.get_infos())

	# Splitting polygons may result in panels slightly overlapping, de-overlap them
	def deoverlap_panels(self):
		for p1 in self.panels:
			for p2 in self.panels:
				if p1 == p2:
					continue

				opanel = p1.overlap_panel(p2)
				if not opanel:
					continue

				if opanel.w() < opanel.h() and p1.r == opanel.r:
					p1.r = opanel.x
					p2.x = opanel.r
					continue

				if opanel.w() > opanel.h() and p1.b == opanel.b:
					p1.b = opanel.y
					p2.y = opanel.b
					continue

		Debug.add_step('Deoverlap panels', self.get_infos())

	# Merge panels that shouldn't have been split (speech bubble diving into a panel)
	def merge_panels(self):
		panels_to_remove = []
		for i, p1 in enumerate(self.panels):
			for j, p2 in enumerate(self.panels[i + 1:]):
				if p1.contains(p2):
					panels_to_remove.append(p2)
					p1 = p1.merge(p2)
				elif p2.contains(p1):
					panels_to_remove.append(p1)
					p2 = p2.merge(p1)

		for p in set(panels_to_remove):
			self.panels.remove(p)

		Debug.add_step('Merge panels', self.get_infos())

	# Find out actual gutters between panels
	def actual_gutters(self, func = min):
		gutters_x = []
		gutters_y = []
		for p in self.panels:
			left_panel = p.find_left_panel()
			if left_panel:
				gutters_x.append(p.x - left_panel.r)

			top_panel = p.find_top_panel()
			if top_panel:
				gutters_y.append(p.y - top_panel.b)

		if not gutters_x:
			gutters_x = [1]
		if not gutters_y:
			gutters_y = [1]

		return {'x': func(gutters_x), 'y': func(gutters_y), 'r': -func(gutters_x), 'b': -func(gutters_y)}

	def max_gutter(self):
		return max(self.actual_gutters().values())

	# Expand panels to their neighbour's edge, or page boundaries
	def expand_panels(self):
		gutters = self.actual_gutters()
		for p in self.panels:
			for d in ['x', 'y', 'r', 'b']:  # expand in all four directions
				newcoord = -1
				neighbour = p.find_neighbour_panel(d)
				if neighbour:
					# expand to that neighbour's edge (minus gutter)
					newcoord = getattr(neighbour, {'x': 'r', 'r': 'x', 'y': 'b', 'b': 'y'}[d]) + gutters[d]
				else:
					# expand to the furthest known edge (frame around all panels)
					min_panel = min(self.panels, key = lambda p: getattr(p, d)) if d in [
						'x', 'y'
					] else max(self.panels, key = lambda p: getattr(p, d))
					newcoord = getattr(min_panel, d)

				if newcoord != -1:
					if d in ['r', 'b'] and newcoord > getattr(p, d) or d in ['x', 'y'] and newcoord < getattr(p, d):
						setattr(p, d, newcoord)

		Debug.add_step('Expand panels', self.get_infos())

	# Fix panels simple sorting (issue #12)
	def fix_panels_numbering(self):
		changes = 1
		while changes:
			changes = 0
			for i, p in enumerate(self.panels):
				neighbours_before = [p.find_top_panel()]
				neighbours_before += p.find_all_right_panels() if self.numbering == "rtl" else p.find_all_left_panels()

				for neighbour in neighbours_before:
					if neighbour is None:
						continue
					neighbour_pos = self.panels.index(neighbour)
					if i < neighbour_pos:
						changes += 1
						self.panels.insert(neighbour_pos, self.panels.pop(i))
						break
				if changes > 0:
					break  # start a new whole loop with reordered panels

		Debug.add_step('Numbering fixed', self.get_infos())

	# group big panels together
	def group_big_panels(self):
		grouped = True
		while grouped:
			grouped = False
			for i, p1 in enumerate(self.panels):
				for p2 in self.panels[i + 1:]:
					p3 = p1.group_with(p2)

					other_panels = [p for p in self.panels if p not in [p1, p2]]
					if p3.bumps_into(other_panels):
						continue

					# are there big segments in this panel?
					segments = []
					for s in self.segments:
						if p3.contains_segment(s) and s.dist() > p3.diagonal().dist() / 5:
							if s not in segments:
								segments.append(s)

					if len(segments) > 0:  # maybe allow a small number of big segments here?
						continue

					self.panels.append(p3)
					self.panels.remove(p1)
					self.panels.remove(p2)
					grouped = True
					break

				if grouped:
					break

		Debug.add_step('Group big panels', self.get_infos())
