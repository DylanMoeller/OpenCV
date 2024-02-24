import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')  # From OpenCV (not my own training file)
eye_cascade = cv.CascadeClassifier('../Cascades/haarcascade_eye.xml')
capture = cv.VideoCapture(0)
capture.set(3, 500)  # Width and height of camera frame
capture.set(4, 500)

while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=25,  # Higher this is, lower false positives
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
        color = frame[y:y + h, x:x + w]
        gray = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255),2)


    cv.imshow('frame', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv.destroyAllWindows()