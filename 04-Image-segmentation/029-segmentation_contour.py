import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/shapes_donut.jpg"
image = cv2.imread(source_input)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

#Grayscale: key step in finding contours! Passing color image will yield error!
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find Canny Edge
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edge', edged)
cv2.waitKey(0)

#Finding contours
#Use a copy of image e.g edged.copy(), since findContours alters the image
#findContours in openCV3 has 3 returns (a boolean indicating if the func was successfully run) while openCV2 has 2 returns.
#hierachy describes the child-parent relationships between contours (i.e contours within contours

"""
Approximation Methods:
Using cv2.CHAIN_APPROX_NONE stores all the boundary points. But we don't necessarily need all bounding points. If the points
form a straight line, we only need the start an ending points of that line
Using cv2.CHAIN_APPROX_SIMPLE instead only provides these start and end points of bounding contours, thus
resulting in much more efficient storage of contour information

Hierachy Type (the first two are the most useful)
cv2.RETR_LIST - retrieve all contours
cv2.RETR_EXTERNAL - retrives external or outer contours only
cv2.RETR_COMP - retrieves all in a 2 level hierachy
cv2.RETR_TREE - retrieves all in full hierachy

Hierachy is stored in following format [Next, Previous, First Child, Parent]
"""

#get external contours only cv2.RETR_EXTERNAL
#_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#get all contours cv2.RETR_LIST
_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', image)
cv2.waitKey(0)

cv2.destroyAllWindows()