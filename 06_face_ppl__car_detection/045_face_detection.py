import cv2
import numpy as np

#Point OpenCV's CascadeClassifier function to where our xml classifier is
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#load image then convert to gray scale
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/Trump.jpg"
image = cv2.imread(source_input)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Classifier returns the ROI of the detected face as a tuple
# Store the top left coordinate and the bottom right coordinates
faces = face_classifier.detectMultiScale(grey, 1.3, 5)

#When no face is detected, face_classifier returns an empty tuple
if faces is ():
    print("No faces found")

# Iterate through faces array and draw a rectangle over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (127, 0, 255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()