import cv2

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)
cv2.imshow('Original', image)
cv2.waitKey()

#use cvtColor to convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey()
cv2.destroyAllWindows()


#FASTER METHOD
img = cv2.imread(source_input, 0) #flag 0 is grayscale
cv2.imshow('Grayscale fast', img)
cv2.waitKey()
cv2.destroyAllWindows()

#grayscale operates much quicker