import cv2 as cv
import numpy as np

def video_demo():
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()

        frame = cv.flip(frame,1)
        # 因为摄像头正对着我们，所以有视角差，可以纠正
        # frame = cv.flip(frame, 1)
        # 摄像头正对，出现视角差，flip镜像调换,还有上下调换1和-1的关系

        # frame 是指图像中的每一帧
        cv.imshow('vidoe',frame)
        c = cv.waitKey(50)
        if c == 27:
            break


def get_image(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)

    pix_data = np.array(image)
    print(pix_data)



print("--------- Python OpenCV Tutorial ---------")
src = cv.imread("./1111.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

cv.imwrite('./new.png',src)
# 保存图片

# 转灰度图像
# gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
# cv.imwrite('./new1.png',gray)
# get_image(src)
# # video_demo()
# cv.waitKey(0)
#
# cv.destroyAllWindows()
# video_demo()