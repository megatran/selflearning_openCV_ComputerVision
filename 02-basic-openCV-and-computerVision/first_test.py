import cv2
import numpy as np

source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
print(source_input)

#load an image using 'imread'
input = cv2.imread(source_input)

#display image variable, use 'imshow'
cv2.imshow('Hello World', input)

"""waitKey allows us to input information when an image is open. By leaving it blank, it will wait for anykey to be
pressed before continuing. By placing numbers (except 0), we can specify a delay for how long
you want the window open (time is in milliseconds)
"""
cv2.waitKey(0)

"""
This closes all windows. Failure to place this will cause the program to hang
"""
cv2.destroyAllWindows()

#(H, W, RGB value)
print(input.shape)
print('Height of image: ', int(input.shape[0]), 'pixels')
print('Width of image: ', int(input.shape[1]), 'pixels')

#use iwrite to specifice the file name and the image to be saved
#cv2.imwrite('output.jpg', input)
#cv2.imwrite('output.png', input)