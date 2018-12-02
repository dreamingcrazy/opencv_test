import cv2 as cv

def get_video():
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()
        cv.imshow('video',frame)



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
    cv.namedWindow('video')
    # get_image(img)
    get_video()

    # 转灰度图
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    cv.imwrite('./gray.png',gray)

    cv.namedWindow('image')
    cv.imshow('image',img)

    cv.namedWindow('gray')
    cv.imshow('gray',gray)

    cv.waitKey()