import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#OpenCV 'split' func slice the image into each color index
B, G, R = cv2.split(image)

print(B.shape)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
cv2.destroyAllWindows()

#remake original image
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

#amplify blue color
amplified = cv2.merge([B+100, G, R])
cv2.imshow("Merge with Blue amplified", amplified)

cv2.waitKey(0)
cv2.destroyAllWindows()