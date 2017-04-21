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



