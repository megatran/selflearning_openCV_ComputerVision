import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)
height, width = image.shape[:2]

#get starting pixel coordinate (top left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

#get ending pixel coordinate (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

#use indexing to crop out rectangle we desire
cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("Original Image", image)
cv2.waitKey(0)

cv2.imshow("Cropped image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()