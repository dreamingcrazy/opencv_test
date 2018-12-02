import cv2 as cv

def get_image(image):
    print(image.dtype)
    print(image.size)
    print(image.shape)
#     shape  height width chanel



if __name__ == '__main__':

    print('*'*100)
    # cv.imshow()
    # cv.imwrite()
    img = cv.imread('./1111.jpg')
    get_image(img)


    # 转灰度图
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    cv.imwrite('./gray.png',gray)

    cv.namedWindow('image')
    cv.imshow('image',img)

    cv.namedWindow('gray')
    cv.imshow('gray',gray)

    cv.waitKey()