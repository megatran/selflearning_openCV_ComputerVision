import cv2
import numpy as np


#Funcs for sorting by position

"""
Image moments help you to calculate some features like center of mass of the object, area of the object etc
From this moments, you can extract useful data like area, centroid etc.
Centroid is given by the relations, Cx=M10/M00 and Cy=M01/M00.
"""

def x_cord_contour(contours):
    #return the x coordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))

def label_contour_center(image, c):
    #Places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    #draw contour number on the image
    cv2.circle(image, (cx, cy), 10, (0,0,255), -1)
    return image

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/bunchofshapes.jpg"
image = cv2.imread(source_input)

#Create a black image with same dimensions as loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))

#Create a copy of our original image
original_image = image.copy()

#greyscale our image
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find canny edges
edged = cv2.Canny(grey, 50, 200)
cv2.imshow('1 - Canny Edges', edged)
cv2.waitKey(0)

#Find contours and print how many were found
_, contours, hierachy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Compute Center of Mass or centroids and draw them on image
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c)

cv2.imshow("Contour Center", image)
cv2.waitKey(0)

#sorted by left to right using x_cord_contour function
contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)

#Label contour left to right
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image, [c], -1, (0,0,255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(original_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Left to Right Contour', original_image)
    cv2.waitKey(0)
    #crop contour. Extract 4 points (starting points x, y) and (w, h) boundaries
    (x, y, w, h) = cv2.boundingRect(c)

    #Crop each contour and save these images
    cropped_contour = original_image[y:y + h, x:x + w]
    image_name = "output_shape_number_" + str(i+1) + ".jpg"
    print(image_name)
    cv2.imwrite(image_name, cropped_contour)

cv2.destroyAllWindows()