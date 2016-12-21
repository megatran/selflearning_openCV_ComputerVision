import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#make image 3/4 of its original size
image_scaled_linear = cv2.resize(image, None, fx=0.75, fy=0.75)
cv2.imshow('Scaling - Linear Interpolation', image_scaled_linear)
cv2.waitKey()

#double size of image using cubic interpolation
image_scaled_cubic = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow("Scaling - Cubic Interpolation", image_scaled_cubic)
cv2.waitKey()

#skew the re-sizing by setting exact dimension
image_scaled_skew = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow("Scaling - Skewed Size", image_scaled_skew)
cv2.waitKey()

cv2.destroyAllWindows()
