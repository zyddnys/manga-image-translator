import math
import cv2 as cv
import numpy as np

from .segment import Segment
from .debug import Debug


class Panel:

	@staticmethod
	def from_xyrb(page, x, y, r, b):
		return Panel(page, xywh = [x, y, r - x, b - y])

	def __init__(self, page, xywh = None, polygon = None, splittable = True):
		self.page = page

		if xywh is None and polygon is None:
			raise Exception('Fatal error: no parameter to define Panel boundaries')

		if xywh is None:
			xywh = cv.boundingRect(polygon)

		self.x = xywh[0]  # panel's left edge
		self.y = xywh[1]  # panel's top edge
		self.r = self.x + xywh[2]  # panel's right edge
		self.b = self.y + xywh[3]  # panel's bottom edge

		self.polygon = polygon
		self.splittable = splittable
		self.segments = None
		self.coverage = None

	def w(self):
		return self.r - self.x

	def h(self):
		return self.b - self.y

	def diagonal(self):
		return Segment((self.x, self.y), (self.r, self.b))

	def wt(self):
		return self.w() / 10
		# wt = width threshold (under which two edge coordinates are considered equal)

	def ht(self):
		return self.h() / 10
		# ht = height threshold

	def to_xywh(self):
		return [self.x, self.y, self.w(), self.h()]

	def __eq__(self, other):
		return all(
			[
				abs(self.x - other.x) < self.wt(),
				abs(self.y - other.y) < self.ht(),
				abs(self.r - other.r) < self.wt(),
				abs(self.b - other.b) < self.ht(),
			]
		)

	def __lt__(self, other):
		# panel is above other
		if other.y >= self.b - self.ht() and other.y >= self.y - self.ht():
			return True

		# panel is below other
		if self.y >= other.b - self.ht() and self.y >= other.y - self.ht():
			return False

		# panel is left from other
		if other.x >= self.r - self.wt() and other.x >= self.x - self.wt():
			return True if self.page.numbering == 'ltr' else False

		# panel is right from other
		if self.x >= other.r - self.wt() and self.x >= other.x - self.wt():
			return False if self.page.numbering == 'ltr' else True

		return True  # should not happen, TODO: raise an exception?

	def __le__(self, other):
		return self.__lt__(other)

	def __gt__(self, other):
		return not self.__lt__(other)

	def __ge__(self, other):
		return self.__gt__(other)

	def area(self):
		return self.w() * self.h()

	def __str__(self):
		return f"{self.x}x{self.y}-{self.r}x{self.b}"

	def __hash__(self):
		return hash(self.__str__())

	def is_small(self, extra_ratio = 1):
		return any(
			[
				self.w() < self.page.img_size[0] * self.page.small_panel_ratio * extra_ratio,
				self.h() < self.page.img_size[1] * self.page.small_panel_ratio * extra_ratio,
			]
		)

	def is_very_small(self):
		return self.is_small(1 / 10)

	def overlap_panel(self, other):
		if self.x > other.r or other.x > self.r:  # panels are left and right from one another
			return None
		if self.y > other.b or other.y > self.b:  # panels are above and below one another
			return None

		# if we're here, panels overlap at least a bit
		x = max(self.x, other.x)
		y = max(self.y, other.y)
		r = min(self.r, other.r)
		b = min(self.b, other.b)

		return Panel(self.page, [x, y, r - x, b - y])

	def overlap_area(self, other):
		opanel = self.overlap_panel(other)
		if opanel is None:
			return 0

		return opanel.area()

	def overlaps(self, other):
		opanel = self.overlap_panel(other)
		if opanel is None:
			return False

		area_ratio = 0.1
		smallest_panel_area = min(self.area(), other.area())

		if smallest_panel_area == 0:  # probably a horizontal or vertical segment
			return True

		return opanel.area() / smallest_panel_area > area_ratio

	def contains(self, other):
		o_panel = self.overlap_panel(other)
		if not o_panel:
			return False

		# self contains other if their overlapping area is more than 50% of other's area
		return o_panel.area() / other.area() > 0.50

	def same_row(self, other):
		above, below = sorted([self, other], key = lambda p: p.y)

		if below.y > above.b:  # stricly above
			return False

		if below.b < above.b:  # contained
			return True

		# intersect
		intersection_y = min(above.b, below.b) - below.y
		min_h = min(above.h(), below.h())
		return min_h == 0 or intersection_y / min_h >= 1 / 3

	def same_col(self, other):
		left, right = sorted([self, other], key = lambda p: p.x)

		if right.x > left.r:  # stricly left
			return False

		if right.r < left.r:  # contained
			return True

		# intersect
		intersection_x = min(left.r, right.r) - right.x
		min_w =  min(left.w(), right.w())
		return min_w == 0 or intersection_x / min_w >= 1 / 3

	def find_top_panel(self):
		all_top = list(filter(lambda p: p.b <= self.y and p.same_col(self), self.page.panels))
		return max(all_top, key = lambda p: p.b) if all_top else None

	def find_bottom_panel(self):
		all_bottom = list(filter(lambda p: p.y >= self.b and p.same_col(self), self.page.panels))
		return min(all_bottom, key = lambda p: p.y) if all_bottom else None

	def find_all_left_panels(self):
		return list(filter(lambda p: p.r <= self.x and p.same_row(self), self.page.panels))

	def find_left_panel(self):
		all_left = self.find_all_left_panels()
		return max(all_left, key = lambda p: p.r) if all_left else None

	def find_all_right_panels(self):
		return list(filter(lambda p: p.x >= self.r and p.same_row(self), self.page.panels))

	def find_right_panel(self):
		all_right = self.find_all_right_panels()
		return min(all_right, key = lambda p: p.x) if all_right else None

	def find_neighbour_panel(self, d):
		return {
			'x': self.find_left_panel,
			'y': self.find_top_panel,
			'r': self.find_right_panel,
			'b': self.find_bottom_panel,
		}[d]()

	def group_with(self, other):
		min_x = min(self.x, other.x)
		min_y = min(self.y, other.y)
		max_r = max(self.r, other.r)
		max_b = max(self.b, other.b)
		return Panel(self.page, [min_x, min_y, max_r - min_x, max_b - min_y])

	def merge(self, other):
		possible_panels = [self]

		# expand self in all four directions where other is
		if other.x < self.x:
			possible_panels.append(Panel.from_xyrb(self.page, other.x, self.y, self.r, self.b))

		if other.r > self.r:
			for pp in possible_panels.copy():
				possible_panels.append(Panel.from_xyrb(self.page, pp.x, pp.y, other.r, pp.b))

		if other.y < self.y:
			for pp in possible_panels.copy():
				possible_panels.append(Panel.from_xyrb(self.page, pp.x, other.y, pp.r, pp.b))

		if other.b > self.b:
			for pp in possible_panels.copy():
				possible_panels.append(Panel.from_xyrb(self.page, pp.x, pp.y, pp.r, other.b))

		# don't take a merged panel that bumps into other panels on page
		other_panels = [p for p in self.page.panels if p not in [self, other]]
		possible_panels = list(filter(lambda p: not p.bumps_into(other_panels), possible_panels))

		# take the largest merged panel
		return max(possible_panels, key = lambda p: p.area()) if len(possible_panels) > 0 else self

	def is_close(self, other):
		c1x = self.x + self.w() / 2
		c1y = self.y + self.h() / 2
		c2x = other.x + other.w() / 2
		c2y = other.y + other.h() / 2

		return all(
			[
				abs(c1x - c2x) <= (self.w() + other.w()) * 0.75,
				abs(c1y - c2y) <= (self.h() + other.h()) * 0.75,
			]
		)

	def bumps_into(self, other_panels):
		for other in other_panels:
			if other == self:
				continue
			if self.overlaps(other):
				return True

		return False

	def contains_segment(self, segment):
		other = Panel.from_xyrb(None, *segment.to_xyrb())
		return self.overlaps(other)

	def get_segments(self):
		if self.segments is not None:
			return self.segments

		self.segments = list(filter(lambda s: self.contains_segment(s), self.page.segments))

		return self.segments

	def split(self):
		if self.splittable is False:
			return None

		split = self._cached_split()

		if split is None:
			self.splittable = False

		return split

	def _cached_split(self):
		if self.polygon is None:
			return None

		if self.is_small(extra_ratio = 2):  # panel should be splittable in two non-small subpanels
			return None

		min_hops = 3
		max_dist_x = int(self.w() / 3)
		max_dist_y = int(self.h() / 3)
		max_diagonal = math.sqrt(max_dist_x**2 + max_dist_y**2)
		dots_along_lines_dist = max_diagonal / 5
		min_dist_between_dots_x = max_dist_x / 10
		min_dist_between_dots_y = max_dist_y / 10

		# Compose modified polygon to optimise splits
		original_polygon = np.copy(self.polygon)
		polygon = np.ndarray(shape = (0, 1, 2), dtype = int, order = 'F')
		intermediary_dots = []
		extra_dots = []

		for i in range(len(original_polygon)):
			j = (i + 1) % len(original_polygon)
			dot1 = tuple(original_polygon[i][0])
			dot2 = tuple(original_polygon[j][0])
			seg = Segment(dot1, dot2)

			# merge nearby dots together
			if seg.dist_x() < min_dist_between_dots_x and seg.dist_y() < min_dist_between_dots_y:
				original_polygon[j][0] = seg.center()
				continue

			polygon = np.append(polygon, [[dot1]], axis = 0)

			# Add dots on *long* edges, by projecting other polygon dots on this segment
			add_dots = []

			# should be splittable in [dot1, dot1b(?), projected_dot3, dot2b(?), dot2]
			if seg.dist() < dots_along_lines_dist * 2:
				continue

			for k, dot3 in enumerate(original_polygon):
				if abs(k - i) < min_hops:
					continue

				projected_dot3 = seg.projected_point(dot3)

				# Segment should be able to contain projected_dot3
				if not seg.may_contain(projected_dot3):
					continue

				# dot3 should be close to current segment − distance(dot3, projected_dot3) should be short
				project = Segment(dot3[0], projected_dot3)
				if project.dist_x() > max_dist_x or project.dist_y() > max_dist_y:
					continue

				# append dot3 as intermediary dot on segment(dot1, dot2)
				add_dots.append(projected_dot3)
				intermediary_dots.append(projected_dot3)

			# Add also a dot near each end of the segment (provoke segment matching)
			alpha_x = math.acos(seg.dist_x(keep_sign = True) / seg.dist())
			alpha_y = math.asin(seg.dist_y(keep_sign = True) / seg.dist())
			dist_x = int(math.cos(alpha_x) * dots_along_lines_dist)
			dist_y = int(math.sin(alpha_y) * dots_along_lines_dist)

			dot1b = (dot1[0] + dist_x, dot1[1] + dist_y)
			# if len(intermediary_dots) == 0 or Segment(dot1b, intermediary_dots[0]).dist() > dots_along_lines_dist:
			add_dots.append(dot1b)
			extra_dots.append(dot1b)

			dot2b = (dot2[0] - dist_x, dot2[1] - dist_y)
			# if len(intermediary_dots) == 0 or Segment(dot2b, intermediary_dots[-1]).dist() > dots_along_lines_dist:
			add_dots.append(dot2b)
			extra_dots.append(dot2b)

			for dot in sorted(add_dots, key = lambda dot: Segment(dot1, dot).dist()):
				polygon = np.append(polygon, [[dot]], axis = 0)

		# Re-merge nearby dots together
		original_polygon = np.copy(polygon)
		polygon = np.ndarray(shape = (0, 1, 2), dtype = int, order = 'F')

		for i in range(len(original_polygon)):
			j = (i + 1) % len(original_polygon)
			dot1 = tuple(original_polygon[i][0])
			dot2 = tuple(original_polygon[j][0])
			seg = Segment(dot1, dot2)

			# merge nearby dots together
			if seg.dist_x() < min_dist_between_dots_x and seg.dist_y() < min_dist_between_dots_y:
				intermediary_dots = [dot for dot in intermediary_dots if dot not in [dot1, dot2]]
				extra_dots = [dot for dot in extra_dots if dot not in [dot1, dot2]]
				original_polygon[j][0] = seg.center()
				continue

			polygon = np.append(polygon, [[dot1]], axis = 0)

		Debug.draw_polygon(polygon)
		Debug.draw_dots(intermediary_dots, Debug.colours['red'])
		Debug.draw_dots(extra_dots, Debug.colours['yellow'])
		Debug.add_image(f"Composed polygon {self} ({len(polygon)} dots, {len(intermediary_dots)} intermediary)")

		# Find dots nearby one another
		nearby_dots = []

		for i in range(len(polygon) - min_hops):
			for j in range(i + min_hops, len(polygon)):
				dot1 = polygon[i][0]
				dot2 = polygon[j][0]
				seg = Segment(dot1, dot2)

				if seg.dist_x() <= max_dist_x and seg.dist_y() <= max_dist_y:
					nearby_dots.append([i, j])

		if len(nearby_dots) == 0:
			return None

		Debug.draw_nearby_dots(polygon, nearby_dots)
		Debug.add_image(f"Nearby dots ({len(nearby_dots)})")

		splits = []
		for dots in nearby_dots:
			poly1len = len(polygon) - dots[1] + dots[0]
			poly2len = dots[1] - dots[0]

			# A panel should have at least three edges
			if min(poly1len, poly2len) <= 2:
				continue

			# Construct two subpolygons by distributing the dots around our nearby dots
			poly1 = np.zeros(shape = (poly1len, 1, 2), dtype = int)
			poly2 = np.zeros(shape = (poly2len, 1, 2), dtype = int)

			x = y = 0
			for i in range(len(polygon)):
				if i <= dots[0] or i > dots[1]:
					poly1[x][0] = polygon[i]
					x += 1
				else:
					poly2[y][0] = polygon[i]
					y += 1

			panel1 = Panel(self.page, polygon = poly1)
			panel2 = Panel(self.page, polygon = poly2)

			if panel1.is_small() or panel2.is_small():
				continue

			if panel1 == self or panel2 == self:
				continue

			if panel1.overlaps(panel2):
				continue

			split_segment = Segment.along_polygon(polygon, dots[0], dots[1])
			split = Split(self, panel1, panel2, split_segment)
			if split not in splits:
				splits.append(split)

		Debug.draw_segments([split.segment for split in splits], Debug.colours['red'], size = 2)
		Debug.add_image(f"Splits ({len(splits)})")

		splits = list(filter(lambda split: split.segments_coverage() > 50 / 100, splits))

		if len(splits) == 0:
			return None

		# return the split that best matches segments (~panel edges)
		best_split = max(splits, key = lambda split: split.covered_dist)

		return best_split


class Split:

	def __init__(self, panel, subpanel1, subpanel2, split_segment):
		self.panel = panel
		self.subpanels = [subpanel1, subpanel2]
		self.segment = split_segment

		self.matching_segments = self.segment.intersect_all(self.panel.get_segments())
		self.covered_dist = sum(map(lambda s: s.dist(), self.matching_segments))

	def __eq__(self, other):
		return self.segment == other.segment

	def segments_coverage(self):
		segment_dist = self.segment.dist()
		return self.covered_dist / segment_dist if segment_dist else 0
