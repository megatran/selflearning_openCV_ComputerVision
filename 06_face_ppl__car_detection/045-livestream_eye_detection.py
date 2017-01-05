import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # convert image to grayscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    detectMultiScale(img, scaleFactor, minNeighbors)
    scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    """
    faces = face_classifier.detectMultiScale(grey, 1.3, 5)
    if faces is ():
        return img

    for (x, y, w, h) in faces:
        x = x-50
        w = w+50
        y = y-50
        h = h+50
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = grey[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)

    """
    The function flip flips the array in one of three different ways (row and column indices are 0-based)
    Vertical flipping of the image (flipCode == 0) to switch between top-left and bottom-left image origin. This is a typical operation in video processing on Microsoft Windows* OS.

    Horizontal flipping of the image with the subsequent horizontal shift and absolute difference calculation to check for a vertical-axis symmetry (flipCode > 0).

    Simultaneous horizontal and vertical flipping of the image with the subsequent shift and absolute difference calculation to check for a central symmetry (flipCode < 0).

    Reversing the order of point arrays (flipCode > 0 or flipCode == 0).
    """
    print("roi_color before flip()")
    print(roi_color)
    roi_color = cv2.flip(roi_color, 1)
    print("ori_color after flilp()")
    print(roi_color)
    return roi_color

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Face extractor', face_detector(frame))
    if cv2.waitKey(1) == 13: #13 is the enter key
        break

cap.release()
cv2.destroyAllWindows()

"""
Tuning Cascade Classifiers

ourClassifier.detectMultiScale(input image, Scale Factor , Min Neighbors)

Scale Factor Specifies how much we reduce the image size each time we scale. E.g. in face detection we typically use 1.3. This means we reduce the image by 30% each time it’s scaled. Smaller values, like 1.05 will take longer to compute, but will increase the rate of detection.

Min Neighbors Specifies the number of neighbors each potential window should have in order to consider it a positive detection. Typically set between 3-6. It acts as sensitivity setting, low values will sometimes detect multiples faces over a single face. High values will ensure less false positives, but you may miss some faces.
"""