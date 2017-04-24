import numpy as np
import random
from util import Edgel, LineSegment
from sklearn.preprocessing import normalize

MIN_INLIERS = 5

def atLine(start, end, p):
	if not start.orientationCompatible(p): return False
	LINE_DIST_THRESHOLD = 0.75
	dist = np.abs(np.cross((end - start).toPoint(), (p - start).toPoint()) / np.linalg.norm((end - start).toPoint()))
	return dist < LINE_DIST_THRESHOLD

def ransac(edgelsList, iteration):
	MIN_INLIERS = 5
	lineSegmentList = []

	while True:
		if len(edgelsList) < MIN_INLIERS: return lineSegmentList
		validSamples = False
		supportInRun = []
		for t in range(50):
			for ti in range(100):
				startP, endP = random.sample(edgelsList, 2)
				if startP.orientationCompatible(endP):
					slope = startP.slope
					supportEdgels = filter(lambda edgel: atLine(startP, endP, edgel), edgelsList)
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
				nslope = -nslope
			nslope = normalize([nslope])[0]
			lineSegmentList.append(LineSegment(start, end, nslope))

			for edgel in supportInRun:
				edgelsList.remove(edgel)
		else:
			break

	return lineSegmentList

def lineCompatible(startSeg, endSeg):
	if startSeg.end.distance(endSeg.start) > 25 * 25:
		return False
	if np.dot(startSeg.slope, endSeg.slope) < 0.99 or np.dot(startSeg.slope, normalize([(endSeg.start - startSeg.end).toTuple()])[0]) < 0.99:
		return False


def mergeLineSegments(lineSegmentList):
	next = [-1 for n in range(len(lineSegmentList))]
	startFlag = [True for n in range(len(lineSegmentList))]
	for i, startSegment in enumerate(lineSegmentList):
		for j, endSegment in enumerate(lineSegmentList):
			if not i == j and lineCompatible(startSegment, endSegment):
				if next[i] >= 0:
					print "Merge one segment to more than one segments!"
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




