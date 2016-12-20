import cv2
import numpy as np

#create a black image
image = np.zeros((512, 512,3), np.uint8)

#make this in black and white
image_bw = np.zeros((512, 512), np.uint8)

cv2.imshow("Black Rectangle (Color)", image)
cv2.imshow("Black Rectangle (B&W)", image_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()

#draw diagonal line
line = np.zeros((512, 512, 3), np.uint8)
cv2.line(line, (0,0), (511, 511), (255, 127, 0), 5)
cv2.imshow("Blue Line", line)
cv2.waitKey(0)
cv2.destroyAllWindows()

#draw rectangle
rect = np.zeros((512, 512, 3), np.uint8)
# thickness -1 fills rectangle, 5 thickness
cv2.rectangle(rect, (100, 100), (300, 250), (127,50,127),-1)
cv2.imshow("Rectangle", rect)
cv2.waitKey(0)
cv2.destroyAllWindows()

#draw circle
circle = np.zeros((512, 512, 3), np.uint8)
cv2.circle(circle, (350,350), 100, (15, 75, 50), 5)
cv2.imshow("Circle", circle)
cv2.waitKey(0)
cv2.destroyAllWindows()

#draw polygon
polygon = np.zeros((512, 512, 3), np.uint8)
#define four points
pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)
#reshape points in form required by polylines
pts = pts.reshape((-1,1,2))
cv2.polylines(polygon, [pts], True, (0,0,255), 3)
cv2.imshow("Polygon", polygon)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Add Text
text = np.zeros((512, 512,3), np.uint8)
cv2.putText(text, 'Hello Nhan', (75, 290), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 180,0), 3)
cv2.imshow("Hello Nhan", text)
cv2.waitKey()
cv2.destroyAllWindows()