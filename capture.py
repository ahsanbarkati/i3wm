import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray,(50,50))
cv2.imwrite('capture.jpg',gray)
cap.release()
