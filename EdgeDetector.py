import numpy as np
from sklearn.preprocessing import normalize
from util import Edgel, LineSegment
import cv2
from copy import deepcopy
import random

class EdgeDetector:
	__HORIZONTAL = 0
	__VERTICAL = 1
	EDGEL_THREDSHOLD = np.array([16 * 16]  * 3)

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
			cv2.rectangle(self.segImage, (top, left), (top + height, left + width), (0, 0, 0))

		edgelList = []
		SCANWIDTH = 5
		S = set()

		for x in range(left, left + width, SCANWIDTH):
			prev1 = self.edgeKernel(x, top - 1, self.__VERTICAL) if top >= 1 else np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for y in range(top, top + height + 1):
				current = self.edgeKernel(x, y, self.__VERTICAL)
				current = (current if oneLessThan(self.EDGEL_THREDSHOLD, current) else np.array([0] * 3))
				# To recognize black/white marker use allLessThan to find local maximun
				# To recognize colorful marker, use oneLessThan?
				if y > top and (((x, y - 1) not in S) and oneLessThan(prev2, prev1) and oneLessThan(current, prev1)):
					edgelList.append(Edgel(x, y - 1, self.calculateSlope(x, y - 1)))
					S.add((x, y - 1))
				prev2 = prev1
				prev1 = current

		for y in range(top, top + height, SCANWIDTH):
			prev1 = self.edgeKernel(left - 1, y, self.__HORIZONTAL) if left >=1 else np.array([0] * 3)
			prev2 = np.array([0] * 3)
			for x in range(left, left + width + 1):
				current = self.edgeKernel(x, y, self.__HORIZONTAL)
				current = (current if allLessThan(self.EDGEL_THREDSHOLD, current) else np.array([0] * 3))
				if x > left and (((x - 1, y) not in S) and oneLessThan(prev2, prev1) and oneLessThan(current, prev1)):
					edgelList.append(Edgel(x - 1, y, self.calculateSlope(x - 1, y)))
					S.add((x - 1, y))
				prev2 = prev1
				prev1 = current
		return edgelList

	def atLine(self, start, end, p):
		if not start.orientationCompatible(p): return False
		LINE_DIST_THRESHOLD = 0.75
		dist = np.abs(np.cross((end - start).toPoint(), (p - start).toPoint()) / np.linalg.norm((end - start).toPoint()))
		return dist < LINE_DIST_THRESHOLD

	def findLine_ransac(self, edgelsList):
		MIN_INLIERS = 5
		lineSegmentList = []

		while True:
			if len(edgelsList) < MIN_INLIERS: return lineSegmentList
			validSamples = False
			supportInRun = []
			for t in range(100):
				for ti in range(100):
					startP, endP = random.sample(edgelsList, 2)
					if startP.orientationCompatible(endP):
						slope = startP.slope
						supportEdgels = filter(lambda edgel: self.atLine(startP, endP, edgel), edgelsList)
						if len(supportEdgels) > len(supportInRun):
							supportInRun = supportEdgels
						break

			if len(supportInRun) >= MIN_INLIERS:
				if np.abs(slope[0]) > np.abs(slope[1]):
					start = min(supportInRun, key = lambda s: s.X)
					end = max(supportInRun, key = lambda s: s.X)
				else:
					start = min(supportInRun, key = lambda s: s.Y)
					end = max(supportInRun, key = lambda s: s.Y)

				nslope = np.array([end.X - start.X, end.Y - start.Y])
				if np.dot(nslope, slope) < 0:
					start, end = end, start
				lineSegmentList.append(LineSegment(start, end))

				for edgel in supportInRun:
					edgelsList.remove(edgel)
			else:
				break

		return lineSegmentList

	def lineCompatible(self, startSeg, endSeg):
		length = startSeg.end.distance(endSeg.start)
		if length > 25:
			return False
		slope = normalize([(endSeg.end - startSeg.start).toPoint()])[0]
		if np.dot(startSeg.slope, endSeg.slope) < 0.95 or np.dot(startSeg.slope, slope) < 0.95:
			return False
		p = startSeg.end.toPoint()
		normal = np.array([slope[1], -slope[0]])
		for step in range(int(length)):
			p = p + slope
			x, y = int(p[0]), int(p[1])
			if allLessThan(self.edgeKernel(x, y, self.__VERTICAL), self.EDGEL_THREDSHOLD / 2) \
				and allLessThan(self.edgeKernel(x, y, self.__HORIZONTAL), self.EDGEL_THREDSHOLD / 2):
				return False
			if np.dot(self.calculateSlope(x, y), slope) < 0.38 \
				and np.dot(self.calculateSlope(int(x + normal[0]), int(y + normal[1])), slope) < 0.38 \
				and np.dot(self.calculateSlope(int(x - normal[0]), int(y - normal[1])), slope) < 0.38:
				return False

		return True

	def mergeLineSegments(self, lineSegmentList):
		next = [-1 for n in range(len(lineSegmentList))]
		startFlag = [True for n in range(len(lineSegmentList))]
		for i, startSegment in enumerate(lineSegmentList):
			for j, endSegment in enumerate(lineSegmentList):
				if not i == j and self.lineCompatible(startSegment, endSegment):
					if next[i] >= 0:
						print "Merge one segment to more than one segments!"
						print lineSegmentList[i].end.toPoint(), lineSegmentList[j].start.toPoint(), lineSegmentList[i].slope, lineSegmentList[j].slope
					else:
						next[i] = j
						startFlag[j] = False

		mergedSegList = []
		for i, startSegment in enumerate(lineSegmentList):
			if startFlag[i]:
				p = i
				while next[p] >= 0:
					p = next[p]
				mergedSegList.append(LineSegment(startSegment.start, lineSegmentList[p].end))
		return mergedSegList

	@classmethod
	def drawLine(self, lineSegmentList, image):
		for seg in lineSegmentList:
			cv2.line(image, tuple(reversed(seg.start.toTuple())), tuple(reversed(seg.end.toTuple())), (255, 255, 0))
			cv2.circle(image, tuple(reversed(seg.start.toTuple())), 3, (0, 255, 0), -1)
			cv2.circle(image, tuple(reversed(seg.end.toTuple())), 3, (0, 255, 0), -1)

	def findSegments(self):
		REGION_WIDTH = 40
		REGION_HEIGHT = 40

		if self.debugMode:
			edgelImage = deepcopy(self.img)
			self.segImage = deepcopy(self.img)
			self.mergedLineImage = deepcopy(self.img)

		lineSegments = []

		for x in range(0, self.img.shape[0] - REGION_WIDTH, REGION_WIDTH):
			for y in range(0, self.img.shape[1] - REGION_HEIGHT, REGION_HEIGHT):
				edgelList = self.findEdgelsInRegion(x, y, REGION_WIDTH, REGION_HEIGHT)

				if self.debugMode:
					for edgel in edgelList:
						cv2.circle(edgelImage, (edgel.Y, edgel.X), 1, (0, 255, 0))

				lineSegmentList = self.findLine_ransac(edgelList)
				lineSegments.extend(lineSegmentList)
				if self.debugMode:
					self.drawLine(lineSegmentList, self.segImage)
					for seg in lineSegmentList:
						print 'seg:', seg.start.toPoint(), seg.start.slope, seg.end.toPoint(), seg.end.slope, seg.slope
		mergedLines = self.mergeLineSegments(lineSegments)

		if self.debugMode:
			self.drawLine(mergedLines, self.mergedLineImage)

			cv2.imwrite('result/edgel.jpg', cv2.resize(edgelImage, (0, 0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
			cv2.imwrite('result/segments.jpg', cv2.resize(self.segImage, (0,0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
			cv2.imwrite('result/mergedLineImage.jpg', cv2.resize(self.mergedLineImage, (0,0), fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC))
		# return edgelList

def allLessThan(arr1, arr2):
	return np.min([e1 < e2 for e1, e2 in zip(arr1, arr2)])

def oneLessThan(arr1, arr2):
	return np.max([e1 < e2 for e1, e2 in zip(arr1, arr2)])