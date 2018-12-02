import cv2 as cv
from PIL import Image as ig
import math
import operator
from _functools import reduce
"""打开摄像头"""
def crame():
    cv.namedWindow('sds')
    capture = cv.VideoCapture(0)
    while(capture.isOpened()):
        ret,frame = capture.read()
        if ret==True:
            k = cv.waitKey(40)
            if k == ord('s') or k == ord('S'):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cv.imwrite('b.jpg', frame)
                save_B()
                break
    capture.release()
"""截取图片，灰度储存"""
def save_A():
    image = cv.imread('a.jpg')

    ccf = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    ret = ccf.detectMultiScale(image=image,scaleFactor=1.1,minNeighbors=3,flags=cv.CASCADE_SCALE_IMAGE)
    for(x,y,h,w) in ret:
        cv.rectangle(image,(x,y),(x+w,x+h),(0,0,255))
        image2 = ig.open("a.jpg")
        img = image2.crop((x,y,x+w,y+h))
        img = img.resize((70,70),ig.ANTIALIAS)
        img.save('./a.jpg')

"""保存B图"""
def save_B():
    image = cv.imread('b.jpg')

    ccf = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    ret = ccf.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=3, flags=cv.CASCADE_SCALE_IMAGE)
    for (x, y, h, w) in ret:
        cv.rectangle(image, (x, y), (x + w, x + h), (0, 0, 255))
        image2 = ig.open("b.jpg")
        img = image2.crop((x, y, x + w, y + h))
        img = img.resize((70, 70), ig.ANTIALIAS)
        img.save('./b.jpg')
        duibi()
"""打开图片与A图片对比"""
def duibi():
    pic1 = ig.open('./a.jpg')
    pic2 = ig.open('./b.jpg')
    h1 = pic1.histogram()
    h2 = pic2.histogram()
    difface = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    print(difface)
"""开启函数"""
if __name__ == '__main__':
    crame()