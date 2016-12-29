import cv2
import numpy as np

#import image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/Sunflowers.jpg"
image = cv2.imread(source_input, cv2.IMREAD_GRAYSCALE)

#Set up detector with default parameter
detector = cv2.SimpleBlobDetector_create()

#Detect blob
keypoints = detector.detect(image)

#Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the cirlce correspond to the size of blob
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

#Show keypoints
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()


