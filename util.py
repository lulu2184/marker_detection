import numpy as np

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

class LineSegment:

	def __init__(self, start, end, slope):
		self.start = start
		self.end = end
		self.slope = slope
