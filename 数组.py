import cv2 as cv

import numpy as np


def change_img(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    for row in range(height):
        for col in range(width):
            for c in range(channel):
                each = image[row,col,c]
                image[row, col, c] = 255-each
    cv.imshow('chage_img',image)


def fun(image):
    des = cv.bitwise_not(image)
    cv.imshow('chage_img', des)
image = cv.imread('./1.jpg')

cv.namedWindow('image',cv.WINDOW_AUTOSIZE)

cv.imshow('image',image)

# change_img(image)
fun(image)

cv.waitKey()

cv.destroyAllWindows()