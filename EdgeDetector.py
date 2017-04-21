import numpy as np
from sklearn.preprocessing import normalize
from util import Edgel, LineSegment
import cv2
from copy import deepcopy
import LineModel

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
		if x + 3 > self.img.shape[0] or y + 3 > self.img.shape[1] or x < 3 or y < 3:
			return np.array([0] * 3)
		if direction == self.__VERTICAL:
			return np.abs(np.dot(W, self.img[x, y-2:y+3]))
		else:
			return np.abs(np.dot(W, self.img[x-2:x+3, y]))

	# Calculate gradient intensity for pixel (x, y)
	# Return: unit vector represented gradient orientation (gx, gy)
	def calculateSlope(self, x, y):
		sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		subRegion = cv2.cvtColor(self.img[x-1:x+2, y-1:y+2], cv2.COLOR_BGR2GRAY)
		return normalize([np.array([np.sum(sobel * subRegion), np.sum(sobel.T * subRegion)], dtype = 'float64')])[0]

	# TODO: Consider about the situation that the region exceed the bound of image
	def findEdgelsInRegion(self, left, top, width, height):
		if self.debugMode:
			cv2.rectangle(self.segImgae, (top, left), (top + height, left + width), (0, 0, 0))

		edgelList = []
		THREDSHOLD = np.array([16 * 16]  * 3)
		SCANWIDTH = 5
		S = set()

		for x in range(left, left + width, SCANWIDTH):
			prev1 = np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for y in range(top, top + height):
				current = self.edgeKernel(x, y, self.__VERTICAL)
				current = (current if allLessThan(THREDSHOLD, current) else np.array([0] * 3))
				# To recognize black/white marker use allLessThan to find local maximun
				# To recognize colorful marker, use oneLessThan?
				if (((x, y - 1) not in S) and oneLessThan(prev2, prev1) and oneLessThan(current, prev1)):
					edgelList.append(Edgel(x, y - 1, self.calculateSlope(x, y - 1)))
					S.add((x, y - 1))
				prev2 = prev1
				prev1 = current

		for y in range(top, top + height, SCANWIDTH):
			prev1 = np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for x in range(left, left + width):
				current = self.edgeKernel(x, y, self.__HORIZONTAL)
				current = (current if allLessThan(THREDSHOLD, current) else np.array([0] * 3))
				if (((x - 1, y) not in S) and oneLessThan(prev2, prev1) and oneLessThan(current, prev1)):
					edgelList.append(Edgel(x - 1, y, self.calculateSlope(x - 1, y)))
					S.add((x - 1, y))
				prev2 = prev1
				prev1 = current
		return edgelList


	def findSegments(self):
		REGION_WIDTH = 40
		REGION_HEIGHT = 40

		if self.debugMode:
			edgelImage = deepcopy(self.img)
			self.segImgae = deepcopy(self.img)

		for x in range(0, self.img.shape[0] - REGION_WIDTH, REGION_WIDTH):
			for y in range(0, self.img.shape[1] - REGION_HEIGHT, REGION_HEIGHT):
				edgelList = self.findEdgelsInRegion(x, y, REGION_WIDTH, REGION_HEIGHT)

				if self.debugMode:
					for edgel in edgelList:
						cv2.circle(edgelImage, (edgel.Y, edgel.X), 1, (0, 255, 0))

				lineSementList = LineModel.ransac(edgelList, iteration = 100)
				if self.debugMode:
					for seg in lineSementList:
						cv2.line(self.segImgae, tuple(reversed(seg.start.toTuple())), tuple(reversed(seg.end.toTuple())), (255, 255, 0))

		if self.debugMode:
			cv2.imwrite('edgel.jpg', cv2.resize(edgelImage, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
			cv2.imwrite('segments.jpg', cv2.resize(self.segImgae, (0,0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
		# return edgelList

def allLessThan(arr1, arr2):
	return np.min([e1 < e2 for e1, e2 in zip(arr1, arr2)])

def oneLessThan(arr1, arr2):
	return np.max([e1 < e2 for e1, e2 in zip(arr1, arr2)])