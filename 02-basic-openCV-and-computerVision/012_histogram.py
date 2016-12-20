import cv2
import numpy as np

from matplotlib import pyplot as plt

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

"""
cv2.calcHist(images, channels, mask, hintSize, ranges[,hist[, accumulate]])

IMAGES: source image of type uint8 or float32. Should be give square brackets, ie, "[img]"
CHANNELS: given in sqrt brackets. Index of channel for which we calculate histogram. For example, if input is grayscale
image, its value is [0]. For color image, you can pass [0],[1], or [2] to calculate historgram of blue, green, or red respectively
 mask: mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of
particular region of image, you have to create a mask image for that give it as mask
HISTSIZE: represents our BIN count. Need to be given in sqrt bracket. For full scale, pass [256]
RANGES: our range! Normally [0,256]

"""
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

#plot histogram. Ravel() flatens image array. Make 2d into 1d
plt.hist(image.ravel(), 256, [0,256])
plt.show()

#viewing separate color channels:
color = ('b', 'g', 'r')

#separate the colors and plot each in Histogram

for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])

plt.show()