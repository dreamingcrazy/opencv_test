# import cv2 as cv
# from numpy import *
# from scipy import *
# import numpy as np
#
# #定义添加椒盐噪声的函数
# def SaltAndPepper(src,percetage):
#     SP_NoiseImg=src
#     SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
#     for i in range(SP_NoiseNum):
#         randX=random.random_integers(0,src.shape[0]-1)
#         randY=random.random_integers(0,src.shape[1]-1)
#         if random.random_integers(0,1)==0:
#             SP_NoiseImg[randX,randY]=0
#         else:
#             SP_NoiseImg[randX,randY]=255
#     return SP_NoiseImg
#
# def video_demo():
#     capture = cv.VideoCapture(0)
#     while True:
#         ret,frame =capture.read()
#         if ret == False:
#             break
#
#
#         hsv = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#         SaltAndPepper_noiseImage = SaltAndPepper(hsv, 0.1)  # 再添加10%的椒盐噪声
#         cv.imshow("Add_SaltAndPepperNoise Image", SaltAndPepper_noiseImage)
#         # cv.addWeighted()
#         # cv.imshow('shsv',hsv)
#         # x = cv.Sobel(hsv, cv.CV_16S, 1, 0)
#         # y = cv.Sobel(hsv, cv.CV_16S, 0, 1)
#         # absX = cv.convertScaleAbs(x)
#         # absY = cv.convertScaleAbs(y)
#         # dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
#         # cv.imshow('dst',dst)
#         # c = cv.waitKey(50)
#         # # cv.destroyAllWindows()
#         # if c == 27:
#         #     break
#         # mask = cv.inRange(hsv,upperb=upper,lowerb=lowerb)
#         # dst = cv.bitwise_and(frame,frame,mask=mask)
#         # element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#         # # cv.imshow('video',frame)
#         # # cv.imshow('mask',mask)
#         # # cv.imshow('dst', dst)
#         #
#         # dilate = cv.dilate(frame, element)
#         #
#         # erode = cv.erode(frame, element)
#         #
#         # # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
#         # result = cv.absdiff(dilate, erode)
#         #
#         # # 上面得到的结果是灰度图，cv2.threshold将其二值化以便更清楚的观察结果
#         # # cv2.threshold(src , thresh, maxval, type[, dst])  返回retval、dst
#         # # cv2.threshold(图像, 阈值  , 最大值, 阈值类型)     返回值类型、返回处理后图像
#         # # 阈值类型：THRESH_BINARY、THRESH_BINARY_INV、THRESH_TRUNC、THRESH_TOZERO、THRESH_TOZERO_INV
#         # retval, result = cv.threshold(result, 40, 255, cv.THRESH_BINARY)
#         #
#         # # 反色，即对二值图每个像素取反
#         # result = cv.bitwise_not(result)
#         # # 显示图像
#         # cv.imshow("result", result)
#         # c=cv.waitKey(50)
#         # if c ==27:
#         #     break
#
#
# video_demo()


import cv2
from PIL import Image as ig

def shibie(img):
    classifier = cv2.CascadeClassifier(
        ".\haarcascades\haarcascade_frontalface_default.xml"
    )
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    cv2.imshow("image", img)  # 显示图像
    c = cv2.waitKey(50)
            # # cv.destroyAllWindows()
    if c == 27:
        return 0
def shibietou(img):
    classifier = cv2.CascadeClassifier(".\haarcascades\haarcascade_frontalface_default.xml")
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    for (x, y, h, w) in faceRects:
        '''这是返回的人脸的坐标以及人脸的宽度和高度'''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
        image2 = ig.open('1.jpg')
        img = image2.crop((x, y, x + w, y + h))
    cv2.imshow("image", img)  # 显示图像
    c = cv2.waitKey(50)
    # # cv.destroyAllWindows()
    if c == 27:
        return 0
def video_demo():
    capture = cv2.VideoCapture(0)
    while True:
        ret,frame =capture.read()
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        s = shibie(frame)
        if ret == False or s==0:
            break
if __name__ == '__main__':
    video_demo()
