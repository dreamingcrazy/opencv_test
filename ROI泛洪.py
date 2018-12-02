import cv2 as cv
import numpy as np

def fill_color(image):
    height,width = image.shape[:2]
    mask = np.zeros([height+2,width+2],np.uint8)
    cv.floodFill(image=image,mask=mask,seedPoint=(50,50),newVal=(150,150,150),loDiff=(10,10,10),upDiff=(30,30,30),flags=cv.FLOODFILL_FIXED_RANGE)
    # cv.floodFill(image=image, mask=mask, seedPoint=(50, 50), newVal=(150, 150, 150), loDiff=(10, 10, 10), upDiff=(30, 30, 30), flags=cv.FLOODFILL_MASK_ONLY)
    # mask的值必须要比这张图像长宽各加2
    cv.imshow('fill',image)

    # image, mask, seedPoint, newVal, loDiff = None, upDiff = None, flags = None
def fill_binary():
    image = np.zeros([400,400,3],np.uint8)
    image[100:300,100:300,:] = 255
    cv.imshow('demo',image)

    mask = np.ones([402,402,1],np.uint8)
    mask[101:301,101:301] = 0
    cv.floodFill(image,mask,(200,200),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow('demo1', image)

if __name__ == '__main__':
    src=cv.imread('./pic.png')
    cv.namedWindow('first_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('first_image', src)
    # fill_color(src)
    # face = src[50:180, 180:300]    #选择200:300行、200:400列区域作为截取对象
    # gray = cv.cvtColor(face, cv.COLOR_RGB2GRAY)  #生成的的灰度图是单通道图像
    # backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  #将单通道图像转换为三通道RGB灰度图，因为只有三通道的backface才可以赋给三通道的src
    # src[50:180, 180:300]  = backface
    # cv.imshow("face", src)
    # fill_color(src)
    # fill_binary()
    cv.waitKey(0)
    cv.destroyAllWindows()
