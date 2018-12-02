import cv2 as cv


def con_demo(image):
    dst = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    ret,bia = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('bia',bia)

    # coloneImage,con,heriach = cv.findContours(bia,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    coloneImage, con, heriach = cv.findContours(bia, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i,conu in enumerate(con):
        cv.drawContours(image,con,i,(0,0,255),-1)
    cv.imshow('dect',image)
if __name__ == '__main__':
    src = cv.imread('./55.jpg')
    # src.resize((300,300))

    cv.namedWindow('image',cv.WINDOW_AUTOSIZE)
    cv.imshow('image', src)

    con_demo(src)

    cv.waitKey(0)
    cv.destroyAllWindows()