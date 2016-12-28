"""
cv2.matchShapes(contour template, contour, method, method parameter)
Output - match value (lower values mean a closer match

- contour template: reference contour that we're trying to find in new image
- contour - the individual contour we are checking against
- method - type of contour matching (1,2,3)
- method parameter - leave alone as 0.0 (not fully utilized in python OpenCV)
"""
import cv2
import numpy as np

# Load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/"
template = cv2.imread(source_input+"4star.jpg", 0)
cv2.imshow("Template", template)
cv2.waitKey()

# Load the target image with the shapes we're trying to match
target = cv2.imread(source_input+"shapestomatch.jpg")
target_grey = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold both image first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_grey, 127, 255, 0)

# Find contours in template
_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# We need to sort the contours by area so that we can remove the largest
# contour which is the image outline
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#We extract the second largest contour which will be our template contour
template_contour = contours[1]

# Extract contours from second target image
_, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # Iterate through each contour in the target image and use cv2.matchShapes() to compare contour shapes
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    #if the match value is less than 0.15, then:
    if (match < 0.15):
        closet_contour = c
    else:
        closet_contour = []

cv2.drawContours(target, [closet_contour], -1, (0,255,0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()