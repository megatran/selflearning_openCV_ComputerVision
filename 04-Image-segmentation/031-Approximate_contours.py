"""
cv2.approxPolyDP(contour, Approximation Accuracy, Closed)
- contour : individual contour we wish to approximate
- Approximation Accuracy : important parameter is determining the accuracy of the approximation. Small values
                            give prices approximations, large values give generic approximations. A good rule
                            of thumb is less than 5% of the contour perimeter
- Closed : a Boolean value that states whether the approximate contour should be open or closed
"""

import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/house.jpg"
image = cv2.imread(source_input)
orig_image = image.copy()
cv2.imshow("Original Image", image)
cv2.waitKey(0)

#Greyscale and binarize
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)

#Find contours
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Iterate through each contour and compute the bounding rectangle
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("Bounding rectangle", orig_image)

cv2.waitKey(0)

#Iterate through each contour and compute the approx contour
for c in contours:
    # calculate accuracy as a percent of the contour perimeter
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0,255,0), 2)
    cv2.imshow('Approx Poly DP', image)

cv2.waitKey(0)
cv2.destroyAllWindows()