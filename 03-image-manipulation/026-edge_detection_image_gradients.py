import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input, 0)

height, width = image.shape

#extract sobel edge

sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow("Original", image)
cv2.waitKey(0)

cv2.imshow("Sobel X", sobel_x)
cv2.waitKey(0)

cv2.imshow("Sobel Y", sobel_y)
cv2.waitKey(0)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_OR', sobel_OR)
cv2.waitKey(0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)

"""
Then we need to provide two values: threshold1 and threshold2. Any gradient value larger than threshold2 is
considered to be an edge. Any value below threshold1 is considered not an edge.
Values in between threshold1 and threshold2 are either classified as edges or non-edges based on how
their intensities are "connected". In this case, any gradient values below 60 are considered non-edges
whereas any values above 120 are considered edges
"""

#Canny Edge Detection uses gradient values as thresholds
#The first threshold gradient
canny = cv2.Canny(image, 20, 170)
cv2.imshow("Canny", canny)
cv2.waitKey(0)

cv2.destroyAllWindows()