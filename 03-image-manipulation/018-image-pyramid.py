import cv2

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(image)

cv2.imshow("Original", image)
cv2.imshow("Smaller", smaller)
cv2.imshow("Larger", larger)

cv2.waitKey()
cv2.destroyAllWindows()