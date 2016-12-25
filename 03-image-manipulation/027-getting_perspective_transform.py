import cv2
import numpy as np
import matplotlib.pyplot as plt

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/scan.jpg"
image = cv2.imread(source_input, 0)

cv2.imshow("Original", image)
cv2.waitKey(0)

#Coordinates of the 4 corners of the original image
points_A = np.float32([[320, 15], [700,215], [85, 610], [530, 780]])

#Coordinates of the 4 corners of the desired output
#use ratio of an A4 paper 1 : 1.41
points_B = np.float32([[0,0], [420, 0], [0,594], [420, 594]])

#Use two sets of four points to compute
#perspective transformation matrix M
M = cv2.getPerspectiveTransform(points_A, points_B)

warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow("Warp Perspective", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()