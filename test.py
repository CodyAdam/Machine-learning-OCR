import sys

import numpy as np
import cv2

im = cv2.imread('training_samples/14.png')
if im.any() == None: print("\n\nImage not found!\n\n")

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)


cv2.imshow('gray', gray)
cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.imshow('thresh', thresh)


while True:
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break