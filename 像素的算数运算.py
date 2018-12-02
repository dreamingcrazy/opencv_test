import cv2 as cv


def add_demo(p1,p2):
    add_info = cv.add(p1,p2)
    cv.imshow('add',add_info)

def jian_demo(p1,p2):
    jian = cv.subtract(p2,p1)
    cv.imshow('jian',jian)
if __name__ == '__main__':
    p1 = cv.imread('./1111.jpg')
    p2 = cv.imread('./1111.jpg')
    cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
    cv.imshow('image1',p1)
    cv.imshow('image2', p2)
    # add_demo(p1,p2)
    jian_demo(p1,p2)
    cv.waitKey(0)
    cv.destroyAllWindows()