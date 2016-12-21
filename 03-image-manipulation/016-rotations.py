import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

height, width = image.shape[:2]

#Divide by two to rotate the image around its center
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

#USING TRANSPOSE TO FULLY ROTATE WITHOUT BLACK SPACE
transposed_image = cv2.transpose(image)

cv2.imshow("Rotated Image - Method 2", transposed_image)
cv2.waitKey()
cv2.destroyAllWindows()