import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#Create a matrix of ones, then multiply it by a scalar of 100
#This gives a matrix with same dimensions of image with all value being 100

M = np.ones(image.shape, dtype="uint8") * 75


#Add this and matrix M
added = cv2.add(image, M)
cv2.imshow("Added", added)

#Can also subtract (decrease in brightness)
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)

cv2.waitKey(0)
cv2.destroyAllWindows()