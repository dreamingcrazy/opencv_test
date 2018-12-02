import cv2 as cv

import numpy as np


def video_demo():
    capture = cv.VideoCapture(0)#定义一个cv
    while True:
        ret,frame =capture.read()#对取到的结果进行读取
        if ret == False:
            break

        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)#HSV转化图片
        lowerb = np.array([26,43,46])#hsv的最小值
        upper = np.array([34,255,255])#最大值

        mask = cv.inRange(hsv,upperb=upper,lowerb=lowerb)#识别转换后的图片的规则，最大值最小值
        """bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
            bitwise_or是对二进制数据进行“或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“或”操作，1|1=1，1|0=0，0|1=0，0|0=0
            bitwise_xor是对二进制数据进行“异或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“异或”操作，1^1=0,1^0=1,0^1=1,0^0=0
            bitwise_not是对二进制数据进行“非”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作，~1=0，~0=1
        """
        dst = cv.bitwise_and(frame,frame,mask=mask)

        cv.imshow('video',frame)
        cv.imshow('mask',mask)
        cv.imshow('dst', dst)
        c=cv.waitKey(50)
        if c ==27:
            break


video_demo()