import os, glob,sys
import cv2
import keras
from keras.models import load_model
import numpy as np

mod=sys.argv[1]
pred=sys.argv[2]
im=cv2.imread(pred)

model = load_model(mod)
x=[]
x.append(np.array(im))
resized_image = cv2.resize(im, (50, 50))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
tempimg=pred.split('.')[0]+'_gray.jpg'
cv2.imwrite(tempimg,gray_image)

im=cv2.imread('image_gray.jpg')
x=[]
x.append(np.array(im))
x=np.array(x)
x=x/255
y=model.predict(x)
print(y)
os.system('rm '+tempimg)
