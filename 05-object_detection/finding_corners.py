import cv2
import numpy as np

"""
Corner matching in images is tolerant of:
- Rotations
- Translation
- Slight photometric changes e.g brightness or affine intensity

It is INTOLERANT OF:
- large changes in intensity or photometric changes
- scaling
"""


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

#IMPROVE CORNER DETECTION USING - Good features to Track
img = cv2.imread(source_input)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Specify the top 50 corners
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 150)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (0,255,0), 2)

cv2.imshow("Corners Found", img)
cv2.waitKey()
cv2.destroyAllWindows()

