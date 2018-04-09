import os, glob
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras import regularizers
from subprocess import call
import sys
mod=sys.argv[1]
model = load_model(mod)
cam = cv2.VideoCapture(0)
cv2.namedWindow("test_image")
img_counter = 0
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        im2=cv2.resize(frame,(50,50))
        im2=cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('temp.jpg',im2)
        im=cv2.imread('temp.jpg')
        x=[]
        x.append(np.array(im))
        x=np.array(x)
        x=x/255
        y=model.predict(x)
        print y
        call(["rm","temp.jpg"])
        img_counter += 1


cam.release()

cv2.destroyAllWindows()