import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#two dimension because grayscale image
#if colored image, use rectangle = np.zeros((300, 300, 3), np.uint8)

square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250,250), 255, -2)
cv2.imshow("Square", square)
cv2.waitKey(0)

#Making an ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150,150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("Ellipse", ellipse)
cv2.waitKey(0)

cv2.destroyAllWindows()