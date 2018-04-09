import numpy as np
import pandas
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.datasets import cifar10
import keras
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras import regularizers
import os, glob
import cv2

l=0
for infile in glob.glob( os.path.join('negatives/', "*.jpg") ):
    l=l+1
for infile in glob.glob( os.path.join('positives/', "*.jpg") ):
    l=l+1
X=[]
Y=np.zeros(l,dtype='int32')
i=0
for infile in glob.glob( os.path.join('negatives/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x=np.array(im)
    X.append(list(x))
    Y[i]=int(0)   
    i=i+1
for infile in glob.glob( os.path.join('positives/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x=np.array(im)
    X.append(list(x))
    Y[i]=int(1)
    i=i+1

X=np.array(X)
print(X.shape)
X=X/255

model = Sequential()
model.add(Conv2D(32, kernel_size=(20, 20), strides=(1, 1),
                 activation='relu',
                 input_shape=[50,50,3],kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(50, activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(1, activation='sigmoid',kernel_initializer='random_uniform'))
model.compile(loss=keras.losses.logcosh,optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=50,verbose=1)
model.save('my_face.h5')

