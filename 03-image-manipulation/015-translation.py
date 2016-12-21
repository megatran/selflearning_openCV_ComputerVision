import cv2
import numpy as np

#load input image
source_input = "/Users/nhant/Google Drive/OnlineLearning/selflearning_CV_with_Python/practice_sources/images/input.jpg"
image = cv2.imread(source_input)

#Store height and width of image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

"""
    | 1 0 Tx |
T=  | 0 1 Ty |

T is translation matrix
"""

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
print(T)

#use warpAffine to transform image using the matrix T
image_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow("Translation", image_translation)
cv2.waitKey()
cv2.destroyAllWindows()

