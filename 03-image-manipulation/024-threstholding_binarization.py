#thresholding is act of converting an image to a binary form
#image need to be converted to greyscale before thresholding

import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/gradient.jpg"
image = cv2.imread(source_input, 0)
cv2.imshow("Original", image)

#Values below 127 goes to 0 (black, everything avobe goes to 255-white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("1 Threshold Binary", thresh1)

#Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("2 Threshold Binary Inverse", thresh2)

#Values above 127 are truncated (held) at 127, the 255 argument is unused
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow("3 Thresh Truncate", thresh3)

#Values below 127 go to 0, above 127 are unchanged
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow("4 Thresh Zero", thresh4)

#Reverse of above. Values below 127 are unchanged, above 127 go to 0
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("5. Thresh ToZero Inv", thresh5)
cv2.waitKey(0)

cv2.destroyAllWindows()
