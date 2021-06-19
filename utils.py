
from typing import List
import numpy as np
import cv2

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
		self.aabb = None

	def get_structure(self) -> List[np.ndarray] :
		p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
		p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
		p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
		p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
		return [p1, p2, p3, p4]

	def valid(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v1 = l1b - l1a
		v2 = l2b - l2a
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product) * 180 / np.pi
		return abs(angle - 90) < 10

	def aspect_ratio(self) -> float :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v1 = l1b - l1a
		v2 = l2b - l2a
		return np.linalg.norm(v2) / np.linalg.norm(v1)

	def width(self) -> int :
		return self.get_aabb().w

	def height(self) -> int :
		return self.get_aabb().h

	def clip(self, width, height) :
		self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
		self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

	def get_aabb(self) -> BBox :
		if self.aabb is not None :
			return self.aabb
		kq = self.pts
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		self.aabb = BBox(min_coord[0], min_coord[1], max_coord[0] - min_coord[0], max_coord[1] - min_coord[1], self.text, self.prob, self.fg_r, self.fg_g, self.fg_b, self.bg_r, self.bg_g, self.bg_b)
		return self.aabb

	def get_transformed_region(self, img, direction, textheight) -> np.ndarray :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)
		src_pts = self.pts.astype(np.float32)
		if direction == 'h' :
			h = textheight
			w = round(textheight / ratio)
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			return region
		elif direction == 'v' :
			w = textheight
			h = round(textheight * ratio)
			dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
			M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			region = cv2.warpPerspective(img, M, (w, h))
			region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
			return region

	def is_axis_aligned(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2 :
			return True
		return False

	def is_approximate_axis_aligned(self) -> bool :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v1 = l1b - l1a
		v2 = l2b - l2a
		e1 = np.array([0, 1])
		e2 = np.array([1, 0])
		unit_vector_1 = v1 / np.linalg.norm(v1)
		unit_vector_2 = v2 / np.linalg.norm(v2)
		if abs(np.dot(unit_vector_1, e1)) < 0.06031 or abs(np.dot(unit_vector_1, e2)) < 0.06031 or abs(np.dot(unit_vector_2, e1)) < 0.06031 or abs(np.dot(unit_vector_2, e2)) < 0.06031 :
			return True
		return False

	def direction(self) -> str :
		[l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.get_structure()]
		v_vec = l1b - l1a
		h_vec = l2b - l2a
		if np.linalg.norm(v_vec) > np.linalg.norm(h_vec) :
			return 'v'
		else :
			return 'h'

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

def quadrilateral_can_merge_region(a: Quadrilateral, b: Quadrilateral, ratio = 1.9, char_gap_tolerance = 0.9, char_gap_tolerance2 = 1.5) -> bool :
	a_aa = a.is_approximate_axis_aligned()
	b_aa = b.is_approximate_axis_aligned()
	if a_aa and b_aa :
		b1 = a.get_aabb()
		b2 = b.get_aabb()
		x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
		x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
		dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
		char_size = min(h1, h2, w1, w2)
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
	if not a_aa and not b_aa :
		# TODO: merge non AA text regions
		pass
	return False
