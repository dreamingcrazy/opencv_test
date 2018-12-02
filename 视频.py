import cv2 as cv
def crame():
    cv.namedWindow('frame')
    capture = cv.VideoCapture(0)
    while(capture.isOpened()):
        ret,frame = capture.read()
        if ret==True:
            cv.imshow('frame',frame)
            k = cv.waitKey(40)
            if k==ord('s') or k==ord('S'):
                cv.imwrite('catch.jpg',frame)

                break
    capture.release()
if __name__ == '__main__':
    crame()