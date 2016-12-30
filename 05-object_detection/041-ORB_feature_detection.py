import cv2
import numpy as np

#import image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create ORB object, we can specify the number of key points we desire
orb = cv2.ORB_create(1000)

# determine keypoints
keypoints = orb.detect(gray, None)

#Obtain the descriptors
keypoints, descriptors = orb.compute(gray, keypoints)
print("Number of keypoints detected", len(keypoints))

# draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints,   image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature method - ORB', image)
cv2.waitKey()
cv2.destroyAllWindows()
