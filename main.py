import cv2
from EdgeDetector import EdgeDetector

img = cv2.imread('FullSizeRender.jpg')
edgeDetector = EdgeDetector(img, True)
edgeDetector.findEdgels()
