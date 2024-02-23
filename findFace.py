import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml') #From OpenCV (not my own training file)
capture = cv.VideoCapture(0)
capture.set(3, 500) #Width and height of camera frame
capture.set(4,500)

while True:
    ret, frame = capture.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(1, 1)
    )
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()