import numpy as np
import cv2 as cv

eye_cascade = cv.CascadeClassifier('../Cascades/haarcascade_eye.xml')

capture = cv.VideoCapture(0)
capture.set(3, 500)  # Width and height of camera frame
capture.set(4, 500)

while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    eyes = eye_cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in eyes:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 10)
        color = frame[y:y+h, x:x+w]

        cv.imshow('Eye', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    capture.release()
    cv.destroyAllWindows()