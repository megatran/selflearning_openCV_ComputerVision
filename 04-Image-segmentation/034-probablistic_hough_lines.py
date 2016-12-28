import cv2
import numpy as np

# Grayscale and Canny Edges extracted
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/soduku.jpg"
image = cv2.imread(source_input)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image, 100, 170, apertureSize=3)

#We use rho and theta accuracies
#However, we specifize a minimum vote (pts along line) of 100
# and Min line length of 5 pixels and max gap between lines of 10 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, 5, 10)
print(lines.shape)

for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("Probabilistic Hough Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()