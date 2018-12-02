import cv2 as cv
from PIL import Image as ig


image = cv.imread('1.jpg')

ccf = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
ret=ccf.detectMultiScale(image=image,scaleFactor=1.1,minNeighbors=3,flags=cv.CASCADE_SCALE_IMAGE)
i = 0
for (x,y,h,w) in ret:

    '''这是返回的人脸的坐标以及人脸的宽度和高度'''
    cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
    image2=ig.open('1.jpg')
    img=image2.crop((x,y,x+w,y+h))
    img = img.resize((50,50),ig.ANTIALIAS)
    print(img)
    img.save('./fix'+str(i) +'.jpg')
    i += 1
# img=cv.imread(img)
# cv.imshow('img',img)
cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
cv.imshow('image',image)

cv.waitKey()
cv.destroyAllWindows()