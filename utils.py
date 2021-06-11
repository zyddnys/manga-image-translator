
import numpy as np

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

	def to_points(self) :
		tl, tr, br, bl = np.array([self.x, self.y]), np.array([self.x + self.w, self.y]), np.array([self.x + self.w, self.y+ self.h]), np.array([self.x, self.y + self.h])
		return tl, tr, br, bl