import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#B, G, R values for the first 0,0 pixel
B, G, R = image[0,0]
print(image.shape) #3 dim
print(B, G, R)

#B, G, R values for the first 10,50 pixel
B, G, R = image[10,50]
print(B, G, R)

#convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape) #only 2 dim now
print(gray_image[10,50]) #only one value now

#COLOR FILTERING IWTH HSV

# H (0-180), S (0-255), V (0-255)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv_image)
cv2.imshow('Hue Channel', hsv_image[:, :, 0])
cv2.imshow('Saturation channel', hsv_image[:, :, 1])
cv2.imshow('Value channel', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()
