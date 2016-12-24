import cv2
import numpy as np

#in cv2.adaptiveThrshold(), block sizes must be odd numbers!

source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/Origin_of_Species.jpg"

image = cv2.imread(source_input, 0)

cv2.imshow("Original", image)
cv2.waitKey(0)

#Values below 127 goes to 0 (black, everything above goes to 255-white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#It's good practice to blur images as it removes noice
image = cv2.GaussianBlur(image, (3,3), 0)

#using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,5)
cv2.imshow("Adaptive Mean Thresholding", thresh)
cv2.waitKey(0)

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Thresholding", thresh)
cv2.waitKey(0)

#Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Gaussian Otsu's Thresholding", th3)
cv2.waitKey(0)