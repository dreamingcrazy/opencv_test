import cv2 as cv
import numpy as np

def demo(image):
    dst=cv.blur(image,(10,1))
    # src, ksize,
    cv.imshow('demo',dst)


def middle(image):
    dst2 = cv.medianBlur(image,5)
    cv.imshow('demo2', dst2)


def self_demo1(image):
    '''自定义模糊'''
    kernel = np.ones([5,5],np.float32)/25
#     除以25防止溢出的问题,因为是0-255，防止计算过后溢出
    dst = cv.filter2D(image,-1,kernel=kernel)
    # src, ddepth默认-1，复制克隆自定义的算子, kernel, dst = None, anchor = 卷积中心, delta = None, borderType：边缘的填充模式
    cv.imshow('self', dst)



def self_demo(image):
    '''自定义模糊'''
    # kernel = np.ones([5,5],np.float32)/25
    kernel = np.array([[1,0,1],[1,3,1],[1,1,1]], np.float32) / 9

    dst = cv.filter2D(image,-1,kernel=kernel)
    # src, ddepth默认-1，复制克隆自定义的算子, kernel, dst = None, anchor = 卷积中心, delta = None, borderType：边缘的填充模式
    cv.imshow('self', dst)
# [[0,-1,0],[-1,5,-1],[0,-1,0]]这个称为算子，锐化算子，这个算子的计算，一是为奇数，二是不是奇数的话，加起来是0或1  其中0是在做边缘梯度，1 是做增强的工作
def self_demo2(image):
    '''自定义模糊'''
    # kernel = np.ones([5,5],np.float32)/25
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32) / 9
#     除以25防止溢出的问题,因为是0-255，防止计算过后溢出
    dst = cv.filter2D(image,-1,kernel=kernel)
    # src, ddepth默认-1，复制克隆自定义的算子, kernel, dst = None, anchor = 卷积中心, delta = None, borderType：边缘的填充模式
    cv.imshow('blur', dst)

    # 边缘保留滤波
    # 模糊和滤波区别

if __name__ == '__main__':
    src = cv.imread('./asd.png')
    src2 = cv.imread('./pic.png')
    cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
    cv.imshow('image', src)
    cv.imshow('image2', src2)
    demo(src)
    middle(src)
    self_demo2(src2)
    cv.waitKey(0)
    cv.destroyAllWindows()