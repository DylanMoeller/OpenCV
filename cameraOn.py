import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while (True):
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(frame, 100,200)
    cv.imshow('frame', frame)
    cv.imshow('gray', gray)
    cv.imshow('edges', edges)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()