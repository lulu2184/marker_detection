import cv2
from EdgeDetector import EdgeDetector

img = cv2.imread('marker.jpg')
# img = cv2.imread('test.png')
edgeDetector = EdgeDetector(img, True)
edgeDetector.findSegments()
