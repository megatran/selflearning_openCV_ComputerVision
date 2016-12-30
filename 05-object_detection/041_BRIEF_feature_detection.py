import cv2
import numpy as np


# DOESN'T WORK :(

#import image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Create FAST detector object
fast = cv2.FastFeatureDetector_create()

#Create BRIEF extractor object
#opencv 2 --> brief = cv2.DescriptorExtractor_create("BRIEF")
#opencv 3
brief = cv2.DescriptorExtractor_create("BRIEF")

#Determine keypoints
keypoints = fast.detect(gray, None)

#Obtain descriptors and new final keypoints using BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)
print("Number of kepoints detected: ", len(keypoints))

# Draw rich keypoints on input image
image = cv2.drawKeypoints(gray, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - BRIEF', image)
cv2.waitKey()
cv2.destroyAllWindows()