import cv2
import numpy as np

#import image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST detector object
fast = cv2.FastFeatureDetector_create()

#Obtain keypoints, by default non max suppression is On
# to turn off, set fast.setBool('nonmaxSuppression', False)

keypoints = fast.detect(gray, None)
print("Number of keypoints detected", len(keypoints))

#Draw rich keypoints on input image
image = cv2.drawKeypoints(gray, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature method - FAST', image)
cv2.waitKey()
cv2.destroyAllWindows()