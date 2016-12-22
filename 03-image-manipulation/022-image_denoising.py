import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/elephant.jpg"
image = cv2.imread(source_input)

"""
Parameters, after None are - the filter strength 'h' (5-10 is a good range)
Next is hForColorComponents, set as same value as h again

cv2.fastNlMeansDenoising(): works with a single grayscale image
cv2.fastNlMeansDenoisingColored(): works with a color image
cv2.fastNlMeansDenoisingMulti(): works with image sequence captured in short period of time (grayscale images)
cv2.fastNlMeansDenoisingColoredMulti(): same as above, but for color images
"""

dst = cv2.fastNlMeansDenoisingColored(image, None, 6,6,7,21)

cv2.imshow("Fast Means Denoising", dst)
cv2.waitKey(0)


cv2.destroyAllWindows()