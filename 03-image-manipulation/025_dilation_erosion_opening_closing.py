"""
Dilation: Adds pixels to the boundaries of objects in an image
Erosion: Remove pixels at the boundaries of objects in an image
Opening: Erosion followed by dilation
Closing: Dilation followed by erosion

"""
import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/opencv_inv.png"
image = cv2.imread(source_input, 0)

cv2.imshow("Original", image)
cv2.waitKey(0)

#define kernel size
kernel = np.ones((5,5), np.uint8)

#erode
#iteration increases the effect
erosion = cv2.erode(image, kernel, iterations= 1)
cv2.imshow("Erosion", erosion)
cv2.waitKey(0)

#dilation
dilation = cv2.dilate(image, kernel, iterations= 1)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)

#opening: good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow("Dilation", opening)
cv2.waitKey(0)

#closing: good for removing noise
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closing)
cv2.waitKey(0)

cv2.destroyAllWindows()