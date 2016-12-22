import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/elephant.jpg"
image = cv2.imread(source_input)

"""
Averaging done by convolving the image with a normalized box filter.
This takes pixels under the box and replaces the central element
Box size needs to odd and positive
"""
blur = cv2.blur(image,(3,3))
cv2.imshow("Averaging", blur)
cv2.waitKey(0)

#Instead of box filter, gaussian kernel
gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow("Gaussian Blurring", gaussian)
cv2.waitKey(0)

"""Takes median of all pixels under kernel area and central
element is replaced with this median value
"""
median = cv2.medianBlur(image, 5)
cv2.imshow("Median Blurring", median)
cv2.waitKey(0)

#Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow("Bilateral Blurring", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()