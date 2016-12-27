import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/hand.jpg"
image = cv2.imread(source_input)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Threshold the image
ret, thresh = cv2.threshold(grey, 176, 255, 0)

#Find Contours
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Sort contours by area and then remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key = cv2.contourArea, reverse=False)[:n]

#Iterate through contours and draw convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0,255,0), 2)
    cv2.imshow('Convex Hull', image)

cv2.waitKey(0)
cv2.destroyAllWindows()