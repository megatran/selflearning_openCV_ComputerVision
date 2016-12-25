import cv2
import numpy as np

#Sketch generation function:
def sketch(image):
    #Convert image to greyscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Clean up image using Gaussian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    #Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    #Do an invert binarize the image
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

# Initialize webcam, cap is object provided by VideoCapture
# It contains a boolean indicating if it was successful (ret)
# It also contains the images collected from the webcam (frame)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Live Sketcher", sketch(frame))
    if cv2.waitKey(1) == 13:  #13 is the Enter key
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()