import numpy as np
from sklearn.preprocessing import normalize
from Edgel import Edgel
import cv2

class EdgeDetector:
	__HORIZONTAL = 0
	__VERTICAL = 1

	def __init__(self, img, debugMode):
		self.img = img
		self.debugMode = debugMode

	# direction: 	0 - horizontal
	#				1 - vertical
	def edgeKernel(self, x, y, direction):
		W = [-3, -5, 0, 5, 3]
		if x + 3 > self.img.shape[0] or y + 3 > self.img.shape[1]:
			return np.array([0] * 3)
		if direction == self.__VERTICAL:
			return np.abs(np.dot(W, self.img[x, y-2:y+3]))
		else:
			return np.abs(np.dot(W, self.img[x-2:x+3, y]))

	# Calculate gradient intensity for pixel (x, y)
	# Return: unit vector represented gradient orientation (gx, gy)
	def calculateSlope(self, x, y):
		sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		subRegion = cv2.cvtColor(self.img[x-1:x+2, y-1:y+2], cv2.COLOR_RGB2GRAY)
		return normalize([np.array([np.sum(sobel * subRegion), np.sum(sobel.T * subRegion)], dtype = 'float64')])[0]

	# TODO: Consider about the situation that the region exceed the bound of image
	def findEdgelsInRegion(self, left, top, width, height):
		edgelList = []
		THREDSHOLD = np.array([16 * 16]  * 3)

		for x in range(left, left + width):
			prev1 = np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for y in range(top, top + height):
				current = self.edgeKernel(x, y, self.__VERTICAL)
				current = (current if allLessThan(THREDSHOLD, current) else np.array([0] * 3))
				if (allLessThan(prev2, prev1) and allLessThan(current, prev1)):
					edgelList.append(Edgel(x, y - 1, self.calculateSlope(x, y - 1)))
					print 'y', x, y - 1, prev2, prev1, current
				prev2 = prev1
				prev1 = current

		for y in range(top, top + height):
			prev1 = np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for x in range(left, left + width):
				current = self.edgeKernel(x, y, self.__HORIZONTAL)
				current = (current if allLessThan(THREDSHOLD, current) else np.array([0] * 3))
				if (allLessThan(prev2, prev1) and allLessThan(current, prev1)):
					edgelList.append(Edgel(x - 1, y, self.calculateSlope(x - 1, y)))
					print 'x', x - 1, y, prev2, prev1, current
				prev2 = prev1
				prev1 = current
		return edgelList


	def findEdgels(self):
		REGION_WIDTH = 5
		REGION_HEIGHT = 5

		if self.debugMode:
			edgelImage = self.img

		for x in range(2, self.img.shape[0] - REGION_WIDTH, REGION_WIDTH):
			for y in range(2, self.img.shape[1] - REGION_HEIGHT, REGION_HEIGHT):
				edgelList = self.findEdgelsInRegion(x, y, REGION_WIDTH, REGION_HEIGHT)
				if self.debugMode:
					for edgel in edgelList:
						cv2.circle(edgelImage, (edgel.Y, edgel.X), 1, (0, 255, 0))

		if self.debugMode:
			cv2.imshow('find edgels', edgelImage)
			cv2.imwrite('edgel.jpg', cv2.resize(edgelImage, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
			cv2.waitKey(0)
		# return edgelList

def allLessThan(arr1, arr2):
	return np.min([e1 < e2 for e1, e2 in zip(arr1, arr2)])