import cv2
import numpy as np

#import image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/chess.jpg"
image = cv2.imread(source_input)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The cornerHarris function requires the array datatype to be float32
gray = np.float32(gray)

harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

# We use dilation of the corner points to enlarge them
kernel = np.ones((7,7), np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)

# Threshold for an optimal value, it may vary depending on the image
image[harris_corners > 0.025*harris_corners.max()] = [255,127,127]

cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()