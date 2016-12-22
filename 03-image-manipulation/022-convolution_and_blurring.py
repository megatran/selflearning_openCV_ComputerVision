import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/elephant.jpg"
image = cv2.imread(source_input)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

#creating 3*3 kernel
kernel_3x3 = np.ones((3,3), np.float32) / 9

#use cv2.filter2D to convolve kernel with image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow("3x3 Kernel Blurring", blurred)
cv2.waitKey(0)

kernel_7x7 = np.ones((7,7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow("7x7 Kernel Blurring", blurred2)
cv2.waitKey(0)

cv2.destroyAllWindows()
