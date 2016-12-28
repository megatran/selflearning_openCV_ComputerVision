import cv2
import numpy as np

# Load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/someshapes.jpg"
image = cv2.imread(source_input)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Identifying Shape", image)
cv2.waitKey(0)

ret, thresh = cv2.threshold(grey, 127,255,1)

#extract contours
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    #get approximate polygons
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

    if len(approx) == 3:
        shape_name = "Triangle"
        cv2.drawContours(image, [cnt], 0, (0,255,0), -1)

        #Find contour center to place text at center
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        print("4 contours: ", x,y,w,h)
        M = cv2.moments(cnt)
        cx = int(M['m10'] /  M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Check to see if 4 side polygon is square or rectangle
        #cv2.boundingRect returns the top left and then width and height
        if abs(w-h) <= 3:
            shape_name = "Square"

            # Find contour center to place text at the center
            cv2.drawContours(image, [cnt], 0, (0,125,255), -1)
            cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        else:
            shape_name = "Rectangle"

            # Find contour center to place text at the center
            cv2.drawContours(image, [cnt], 0, (0,0,255), -1)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

    elif len(approx) == 10:
        shape_name = "Star"
        cv2.drawContours(image, [cnt], 0, (255, 255, 0), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

    elif len(approx) >= 15:
        shape_name = "Circle"
        cv2.drawContours(image, [cnt], 0, (0,255,255), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

    cv2.imshow("Identifying Shapes", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
