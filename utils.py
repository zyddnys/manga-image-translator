
from typing import List
import numpy as np
import cv2
import functools
import shapely
from shapely.geometry import Polygon, MultiPoint
from PIL import Image

def convert_img(img) :
	if img.mode == 'RGBA' :
		# from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		alpha_ch = img.split()[3]
		background.paste(img, mask = alpha_ch)  # 3 is the alpha channel
		return background, alpha_ch
	elif img.mode == 'P' :
		img = img.convert('RGBA')
		img.load()  # needed for split()
		background = Image.new('RGB', img.size, (255, 255, 255))
		alpha_ch = img.split()[3]
		background.paste(img, mask = alpha_ch)  # 3 is the alpha channel
		return background, alpha_ch
	else :
		return img.convert('RGB'), None

def resize_keep_aspect(img, size) :
	ratio = (float(size)/max(img.shape[0], img.shape[1]))
	new_width = round(img.shape[1] * ratio)
	new_height = round(img.shape[0] * ratio)
	return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR_EXACT)
	
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

class BBox(object) :
	def __init__(self, x: int, y: int, w: int, h: int, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0) :
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.text = text
		self.prob = prob
		self.fg_r = fg_r
		self.fg_g = fg_g
		self.fg_b = fg_b
		self.bg_r = bg_r
		self.bg_g = bg_g
		self.bg_b = bg_b

	def width(self) :
		return self.w

	def height(self) :
		return self.h

	def to_points(self) :
		tl, tr, br, bl = np.array([self.x, self.y]), np.array([self.x + self.w, self.y]), np.array([self.x + self.w, self.y+ self.h]), np.array([self.x, self.y + self.h])
		return tl, tr, br, bl
		
class Quadrilateral(object) :
	def __init__(self, pts: np.ndarray, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0) :
		self.pts = pts
		self.text = text
		self.prob = prob
		self.fg_r = fg_r
		self.fg_g = fg_g
		self.fg_b = fg_b
		self.bg_r = bg_r
		self.bg_g = bg_g
		self.bg_b = bg_b
		self.assigned_direction = None

	@functools.cached_property
	def structure(self) -> List[np.ndarray] :
		p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
		p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
		p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
		p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
		return [p1, p2, p3, p4]

	@functools.cached_property
	def valid(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product) * 180 / np.pi
		return abs(angle - 90) < 10

	@functools.cached_property
	def aspect_ratio(self) -> float :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		return np.linalg.norm(v2) / np.linalg.norm(v1)

	@functools.cached_property
	def font_size(self) -> float :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		return min(np.linalg.norm(v2), np.linalg.norm(v1))

	def width(self) -> int :
		return self.aabb.w

	def height(self) -> int :
		return self.aabb.h

	def clip(self, width, height) :
		self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
		self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

	@functools.cached_property
	def points(self) :
		ans = [a.astype(np.float32) for a in self.structure]
		return [Point(a[0], a[1]) for a in ans]

	@functools.cached_property
	def aabb(self) -> BBox :
		kq = self.pts
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		return BBox(min_coord[0], min_coord[1], max_coord[0] - min_coord[0], max_coord[1] - min_coord[1], self.text, self.prob, self.fg_r, self.fg_g, self.fg_b, self.bg_r, self.bg_g, self.bg_b)

	def get_transformed_region(self, img, direction, textheight) -> np.ndarray :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)
		src_pts = self.pts.astype(np.float32)
		self.assigned_direction = direction
		if direction == 'h' :
			h = int(textheight)
			w = int(round(textheight / ratio))
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			return region
		elif direction == 'v' :
			w = int(textheight)
			h = int(round(textheight * ratio))
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
			return region

	@functools.cached_property
	def is_axis_aligned(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2 :
			return True
		return False

	@functools.cached_property
	def is_approximate_axis_aligned(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 0.05 or abs(np.dot(unit_vector_1, e2)) < 0.05 or abs(np.dot(unit_vector_2, e1)) < 0.05 or abs(np.dot(unit_vector_2, e2)) < 0.05 :
			return True
		return False

	@functools.cached_property
	def direction(self) -> str :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		if np.linalg.norm(v_vec) > np.linalg.norm(h_vec) :
			return 'v'
		else :
			return 'h'

	@functools.cached_property
	def cosangle(self) -> float :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
		v1 = l1b - l1a
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		return np.dot(unit_vector_1, e2)

	@functools.cached_property
	def angle(self) -> float :
		return np.fmod(np.arccos(self.cosangle) + np.pi, np.pi)

	@functools.cached_property
	def centroid(self) -> np.ndarray :
		return np.average(self.pts, axis = 0)

	def distance_to_point(self, p: np.ndarray) -> float :
		d = 1.0e20
		for i in range(4) :
			d = min(d, distance_point_point(p, self.pts[i]))
			d = min(d, distance_point_lineseg(p, self.pts[i], self.pts[(i + 1) % 4]))
		return d

	@functools.cached_property
	def polygon(self) -> Polygon :
		return MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(self.pts[2]), tuple(self.pts[3])]).convex_hull

	@functools.cached_property
	def area(self) -> float :
		return self.polygon.area

	def poly_distance(self, other) -> float :
		return self.polygon.distance(other.polygon)

	def distance(self, other, rho = 0.5) -> float :
		return self.distance_impl(other, rho)# + 1000 * abs(self.angle - other.angle)

	def distance_impl(self, other, rho = 0.5) -> float :
		assert self.assigned_direction == other.assigned_direction
		#return gjk_distance(self.points, other.points)
		# b1 = self.aabb
		# b2 = b2.aabb
		# x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
		# x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
		# return rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
		pattern = ''
		if self.assigned_direction == 'h' :
			pattern = 'h_left'
		else :
			pattern = 'v_top'
		fs = max(self.font_size, other.font_size)
		if self.assigned_direction == 'h' :
			poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[3]), tuple(other.pts[0]), tuple(other.pts[3])]).convex_hull
			poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[1]), tuple(other.pts[2]), tuple(other.pts[1])]).convex_hull
			poly3 = MultiPoint([
				tuple(self.structure[0]),
				tuple(self.structure[1]),
				tuple(other.structure[0]),
				tuple(other.structure[1])
			]).convex_hull
			dist1 = poly1.area / fs
			dist2 = poly2.area / fs
			dist3 = poly3.area / fs
			if dist1 < fs * rho :
				pattern = 'h_left'
			if dist2 < fs * rho and dist2 < dist1 :
				pattern = 'h_right'
			if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2 :
				pattern = 'h_middle'
			if pattern == 'h_left' :
				return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
			elif pattern == 'h_right' :
				return dist(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
			else :
				return dist(self.structure[0][0], self.structure[0][1], other.structure[0][0], other.structure[0][1])
		else :
			poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(other.pts[0]), tuple(other.pts[1])]).convex_hull
			poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[3]), tuple(other.pts[2]), tuple(other.pts[3])]).convex_hull
			dist1 = poly1.area / fs
			dist2 = poly2.area / fs
			if dist1 < fs * rho :
				pattern = 'v_top'
			if dist2 < fs * rho and dist2 < dist1 :
				pattern = 'v_bottom'
			if pattern == 'v_top' :
				return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
			else :
				return dist(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

def dist(x1, y1, x2, y2) :
	return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
	left = x2b < x1
	right = x1b < x2
	bottom = y2b < y1
	top = y1b < y2
	if top and left:
		return dist(x1, y1b, x2b, y2)
	elif left and bottom:
		return dist(x1, y1, x2b, y2b)
	elif bottom and right:
		return dist(x1b, y1, x2, y2b)
	elif right and top:
		return dist(x1b, y1b, x2, y2)
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	else:             # rectangles intersect
		return 0

def distance_point_point(a: np.ndarray, b: np.ndarray) -> float :
	return np.linalg.norm(a - b)

# from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def distance_point_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray) :
	x = p[0]
	y = p[1]
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]
	A = x - x1
	B = y - y1
	C = x2 - x1
	D = y2 - y1

	dot = A * C + B * D
	len_sq = C * C + D * D
	param = -1
	if len_sq != 0 :
		param = dot / len_sq

	if param < 0 :
		xx = x1
		yy = y1
	elif param > 1 :
		xx = x2
		yy = y2
	else :
		xx = x1 + param * C
		yy = y1 + param * D

	dx = x - xx
	dy = y - yy
	return np.sqrt(dx * dx + dy * dy)


def quadrilateral_can_merge_region(a: Quadrilateral, b: Quadrilateral, ratio = 1.9, discard_connection_gap = 5, char_gap_tolerance = 0.6, char_gap_tolerance2 = 1.5, font_size_ratio_tol = 1.5) -> bool :
	b1 = a.aabb
	b2 = b.aabb
	char_size = min(a.font_size, b.font_size)
	x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
	x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
	dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
	if dist > discard_connection_gap * char_size :
		return False
	if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol :
		return False
	a_aa = a.is_approximate_axis_aligned
	b_aa = b.is_approximate_axis_aligned
	if a_aa and b_aa :
		if dist < char_size * char_gap_tolerance :
			if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2 :
				return True
			if w1 > h1 * ratio and h2 > w2 * ratio :
				return False
			if w2 > h2 * ratio and h1 > w1 * ratio :
				return False
			if w1 > h1 * ratio or w2 > h2 * ratio : # h
				return abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
			elif h1 > w1 * ratio or h2 > w2 * ratio : # v
				return abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
			return False
		else :
			return False
	if True:#not a_aa and not b_aa :
		if abs(a.angle - b.angle) < 15 * np.pi / 180 :
			fs_a = a.font_size
			fs_b = b.font_size
			fs = min(fs_a, fs_b)
			if a.poly_distance(b) > fs * char_gap_tolerance2 :
				return False
			if abs(fs_a - fs_b) / fs > 0.25 :
				return False
			return True
	return False

def quadrilateral_can_merge_region_coarse(a: Quadrilateral, b: Quadrilateral, discard_connection_gap = 2, font_size_ratio_tol = 0.7) -> bool :
	if a.assigned_direction != b.assigned_direction :
		return False
	if abs(a.angle - b.angle) > 15 * np.pi / 180 :
		return False
	fs_a = a.font_size
	fs_b = b.font_size
	fs = min(fs_a, fs_b)
	if abs(fs_a - fs_b) / fs > font_size_ratio_tol :
		return False
	fs = max(fs_a, fs_b)
	dist = a.poly_distance(b)
	if dist > discard_connection_gap * fs :
		return False
	return True

def findNextPowerOf2(n):
	i = 0
	while n != 0 :
		i += 1
		n = n >> 1
	return 1 << i

class Point :
	def __init__(self, x = 0, y = 0) :
		self.x = x
		self.y = y
	
	def length2(self) -> float :
		return self.x * self.x + self.y * self.y

	def length(self) -> float :
		return np.sqrt(self.length2())

	def __str__(self) :
		return f'({self.x}, {self.y})'

	def __add__(self, other):
		x = self.x + other.x
		y = self.y + other.y
		return Point(x, y)

	def __sub__(self, other):
		x = self.x - other.x
		y = self.y - other.y
		return Point(x, y)

	def __mul__(self, other):
		if isinstance(other, Point) :
			return self.x * other.x + self.y * other.y
		else :
			return Point(self.x * other, self.y * other)

	def __truediv__(self, other):
		return self.x * other.y - self.y * other.x

	def neg(self) :
		return Point(-self.x, -self.y)

	def normalize(self) :
		return self * (1. / self.length())

def center_of_points(pts: List[Point]) -> Point :
	ans = Point()
	for p in pts :
		ans.x += p.x
		ans.y += p.y
	ans.x /= len(pts)
	ans.y /= len(pts)
	return ans

def support_impl(pts: List[Point], d: Point) -> Point :
	dist = -1.0e-20
	ans = pts[0]
	for p in pts :
		proj = p * d
		if proj > dist :
			dist = proj
			ans = p
	return ans

def support(a: List[Point], b: List[Point], d: Point) -> Point :
	return support_impl(a, d) - support_impl(b, d.neg())

def cross(a: Point, b: Point, c: Point) -> Point :
	return b * (a * c) - a * (b * c)

def closest_point_to_origin(a: Point, b: Point) -> Point :
	da = a.length()
	db = b.length()
	dist = abs(a / b) / (a - b).length()
	ab = b - a
	ba = a - b
	ao = a.neg()
	bo = b.neg()
	if ab * ao > 0 and ba * bo > 0 :
		return cross(ab, ao, ab).normalize() * dist
	return a.neg() if da < db else b.neg()

def dcmp(a) -> bool :
	if abs(a) < 1e-8 :
		return False
	return True

def gjk_distance(s1: List[Point], s2: List[Point]) -> float :
	d = center_of_points(s2) - center_of_points(s1)
	a = support(s1, s2, d)
	b = support(s1, s2, d.neg())
	d = closest_point_to_origin(a, b)
	s = [a, b]
	for _ in range(8) :
		c = support(s1, s2, d)
		a = s.pop()
		b = s.pop()
		da = d * a
		db = d * b
		dc = d * c
		if not dcmp(dc - da) or not dcmp(dc - db) :
			return d.length()
		p1 = closest_point_to_origin(a, c)
		p2 = closest_point_to_origin(b, c)
		if p1.length2() < p2.length2() :
			s.append(a)
			d = p1
		else :
			s.append(b)
			d = p2
		s.append(c)
	return 0

def main() :
	s1 = [Point(0, 0), Point(0, 2), Point(2, 2), Point(2, 0)]
	offset = 0
	s2 = [Point(1 + offset, 1 + offset), Point(1 + offset, 3 + offset), Point(3 + offset, 3 + offset + 1.5), Point(3 + offset + 1.5, 3 + offset), Point(3 + offset, 1 + offset)]
	print(gjk_distance(s1, s2))

if __name__ == '__main__' :
	main()

