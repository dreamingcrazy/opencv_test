import cv2 as cv

import numpy as np

def demo(image):
    image1 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image2 = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image3 = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow('gray',image1)
    cv.imshow('hsv', image2)
    cv.imshow('crcb', image3)




image = cv.imread('./pic.png')
cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
cv.imshow('image',image)
demo(image)
cv.waitKey()
cv.destroyAllWindows()