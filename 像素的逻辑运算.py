import cv2 as cv
import numpy as np

def logic(p1,p2):
    info = cv.bitwise_or(p1,p2)
    cv.imshow('and',info)



def bright(image,alp,gama):
    '''实现图片的对比度以及亮度的问题'''
    heigt,width,channel = image.shape
    mask = np.zeros([heigt,width,channel],dtype=image.dtype)
    dst = cv.addWeighted(image,alp,mask,1-alp,gama)
    cv.imshow('bright',dst)
    # src1, alpha, src2, beta, gamma, dst = None, dtype = None
if __name__ == '__main__':
    p1 = cv.imread('./a.jpg')
    p2 = cv.imread('./b.jpg')
    p3 = cv.imread('./1111.jpg')
    cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
    # cv.imshow('image',p1)
    # cv.imshow('image2', p2)
    cv.imshow('image', p3)

    # logic(p1,p2)

    bright(p3,1.6,20)
    cv.waitKey(0)
    cv.destroyAllWindows()