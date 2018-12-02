import cv2
cv2.CascadeClassifier()


cammer = cv2.VideoCapture(0)
fps=30
size = (int(cammer.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cammer.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videowrite = cv2.VideoWriter('AA.avi',cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)
success,frame = cammer.read()
numframe = fps*10-1
while success and numframe >0:
    videowrite.write(frame)
    success, frame = cammer.read()
    numframe -= 1
cammer.release()
