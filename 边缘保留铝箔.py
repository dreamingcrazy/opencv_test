import cv2 as cv


def bi_demo(image):
    # 高斯双边
    dst = cv.bilateralFilter(image,0,100,15)
    cv.imshow('dst',dst)
    dst = cv.bilateralFilter(dst, 0, 40, 5)
    cv.imshow('dst2', dst)



def bi_2_demo(image):
    '''均值迁移'''
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow('dst', dst)
if __name__ == '__main__':
    src = cv.imread('./1.jpg')
    # src.resize((300,300))

    cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
    cv.imshow('image', src)

    bi_2_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()