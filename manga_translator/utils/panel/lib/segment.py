import math
import numpy as np


class Segment:

	def __init__(self, a, b):
		self.a = (int(a[0]), int(a[1]))
		self.b = (int(b[0]), int(b[1]))

		for dot in [self.a, self.b]:
			if len(dot) != 2:
				raise Exception(f"Creating a segment with more or less than two dots: Segment({a}, {b})")
			if type(dot[0]) != int or type(dot[1]) != int:
				raise Exception(f"Creating a segment with non-dots: Segment({a}, {b})")

	def __str__(self):
		return f"({self.a}, {self.b})"

	def __eq__(self, other):
		return any([
			self.a == other.a and self.b == other.b,
			self.a == other.b and self.b == other.a,
		])

	def dist(self):
		return math.sqrt(self.dist_x()**2 + self.dist_y()**2)

	def dist_x(self, keep_sign = False):
		dist = self.b[0] - self.a[0]
		return dist if keep_sign else abs(dist)

	def dist_y(self, keep_sign = False):
		dist = self.b[1] - self.a[1]
		return dist if keep_sign else abs(dist)

	def left(self):
		return min(self.a[0], self.b[0])

	def top(self):
		return min(self.a[1], self.b[1])

	def right(self):
		return max(self.a[0], self.b[0])

	def bottom(self):
		return max(self.a[1], self.b[1])

	def to_xyrb(self):
		return [self.left(), self.top(), self.right(), self.bottom()]

	def center(self):
		return (
			int(self.left() + self.dist_x() / 2),
			int(self.top() + self.dist_y() / 2),
		)

	def may_contain(self, dot):
		return all([
			dot[0] >= self.left(),
			dot[0] <= self.right(),
			dot[1] >= self.top(),
			dot[1] <= self.bottom(),
		])

	def intersect(self, other):
		gutter = max(self.dist(), other.dist()) * 5 / 100

		# angle too big ?
		if not self.angle_ok_with(other):
			return None

		# from here, segments are almost parallel

		# segments are apart ?
		if any(
			[
				self.right() < other.left() - gutter,  # self left from other
				self.left() > other.right() + gutter,  # self right from other
				self.bottom() < other.top() - gutter,  # self above other
				self.top() > other.bottom() + gutter,  # self below other
			]
		):
			return None

		projected_c = self.projected_point(other.a)
		dist_c_to_ab = Segment(other.a, projected_c).dist()

		projected_d = self.projected_point(other.b)
		dist_d_to_ab = Segment(other.b, projected_d).dist()

		# segments are a bit too far from each other
		if (dist_c_to_ab + dist_d_to_ab) / 2 > gutter:
			return None

		# segments overlap, or one contains the other
		#  A----B
		#     C----D
		# or
		#  A------------B
		#      C----D
		sorted_dots = sorted([self.a, self.b, other.a, other.b], key = sum)
		middle_dots = sorted_dots[1:3]
		b, c = middle_dots

		return Segment(b, c)

	def union(self, other):
		intersect = self.intersect(other)
		if intersect is None:
			return None

		dots = [tuple(self.a), tuple(self.b), tuple(other.a), tuple(other.b)]
		dots.remove(tuple(intersect.a))
		dots.remove(tuple(intersect.b))
		return Segment(dots[0], dots[1])

	def angle_with(self, other):
		return math.degrees(abs(self.angle() - other.angle()))

	def angle_ok_with(self, other):
		angle = self.angle_with(other)
		return angle < 10 or abs(angle - 180) < 10

	def angle(self):
		return math.atan(self.dist_y() / self.dist_x()) if self.dist_x() != 0 else math.pi / 2

	def intersect_all(self, segments):
		segments_match = []
		for segment in segments:
			s3 = self.intersect(segment)
			if s3 is not None:
				segments_match.append(s3)

		return Segment.union_all(segments_match)

	@staticmethod
	def along_polygon(polygon, i, j):
		dot1 = polygon[i][0]
		dot2 = polygon[j][0]
		split_segment = Segment(dot1, dot2)

		while True:
			i = (i - 1) % len(polygon)
			add_segment = Segment(polygon[i][0], polygon[(i + 1) % len(polygon)][0])
			if add_segment.angle_ok_with(split_segment):
				split_segment = Segment(add_segment.a, split_segment.b)
			else:
				break

		while True:
			j = (j + 1) % len(polygon)
			add_segment = Segment(polygon[(j - 1) % len(polygon)][0], polygon[j][0])
			if add_segment.angle_ok_with(split_segment):
				split_segment = Segment(split_segment.a, add_segment.b)
			else:
				break

		return split_segment

	@staticmethod
	def union_all(segments):
		unioned_segments = True
		while unioned_segments:
			unioned_segments = False
			dedup_segments = []
			used = []
			for i, s1 in enumerate(segments):
				for s2 in segments[i + 1:]:
					if s2 in used:
						continue

					s3 = s1.union(s2)
					if s3 is not None:
						unioned_segments = True
						dedup_segments += [s3]
						used.append(s1)
						used.append(s2)
						break

				if s1 not in used:
					dedup_segments += [s1]

			segments = dedup_segments

		return dedup_segments

	def projected_point(self, p):
		a = np.array(self.a)
		b = np.array(self.b)
		p = np.array(p)
		ap = p - a
		ab = b - a
		if ab[0] == 0 and ab[1] == 0:
			return a
		result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
		return (round(result[0]), round(result[1]))
