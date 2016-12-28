"""
cv2.HoughLines(binarized/threshold image, rho accuracy, theta accuracy, threshold)
Threshold here is the minimum vote for it to be considered a line
"""
import cv2
import numpy as np
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/soduku.jpg"
image = cv2.imread(source_input)

#Greyscale and Canny Edge extracted
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grey, 100, 170, apertureSize= 3)

# Run HoughLines using a rho accuracy of 1 pixel
# Theta accuracy of np.pi / 180 which is 1 degree
# Our line threshold is set to 240 (number of points on line)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)

# We iterate through each line and convert it to the format
# required by cv2.lines(i.e requiring end points)
print(len(lines))
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("Hough Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()