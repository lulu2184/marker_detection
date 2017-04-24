import numpy as np
from sklearn.preprocessing import normalize

ORI_COMP_THREDSHOLD = 0.38

class Edgel:
	X = 0
	Y = 0

	def __init__(self, x, y, slope):
		self.X = x
		self.Y = y
		self.slope = slope

	def __sub__(self, other):
		return Edgel(self.X - other.X, self.Y - other.Y, self.slope)

	def toPoint(self):
		return np.array([self.X, self.Y])

	def toTuple(self):
		return (self.X, self.Y)

	def orientationCompatible(self, other):
		return np.dot(self.toPoint(), other.toPoint()) > ORI_COMP_THREDSHOLD

	def distance(self, other):
		return np.linalg.norm((other - self).toPoint())

class LineSegment:

	#start, end: Edgel
	def __init__(self, start, end, slope = None):
		self.start = start
		self.end = end
		if slope is None:
			self.slope = normalize([np.array([end.X - start.X, end.Y - start.Y])])[0]
		else:
			self.slope = slope
