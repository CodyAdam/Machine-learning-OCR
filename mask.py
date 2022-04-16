import cv2
import numpy as np

img = cv2.imread('training_samples/1-20.png', 0)

blur = cv2.medianBlur(img, 5)

_, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
th1 = cv2.dilate(th1, (np.ones((3, 3), np.uint8)))
cv2.imshow("img", img)
cv2.imshow("blur", blur)
cv2.imshow("th1", th1)
cv2.waitKey(0)