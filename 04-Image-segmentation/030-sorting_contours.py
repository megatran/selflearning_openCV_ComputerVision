import cv2
import numpy as np

#Func to display contour area
def get_contour_areas(contours):
    #return the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/bunchofshapes.jpg"
image = cv2.imread(source_input)
cv2.imshow("0 - Original Image", image)
cv2.waitKey(0)

#Create a black image with same dimensions as loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))

#Create a copy of our original image
original_image = image

#greyscale our image
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find canny edges
edged = cv2.Canny(grey, 50, 200)
cv2.imshow('1 - Canny Edges', edged)
cv2.waitKey(0)

#Find contours and print how many were found
_, contours, hierachy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

#Draw all contours over blank image
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('3 - All Contours', image)
cv2.waitKey(0)

print("Contour Areas before sorting: ", get_contour_areas(contours))

#sort contours large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print("Contour Areas after sorting", get_contour_areas(sorted_contours))

#Iterate over contours and draw one at a time
for cnt in sorted_contours:
    cv2.drawContours(original_image, [cnt], -1, (255, 0, 0), 3)
    cv2.waitKey(0)
    cv2.imshow('Contours by area', original_image)

cv2.waitKey(0)

cv2.destroyAllWindows()
