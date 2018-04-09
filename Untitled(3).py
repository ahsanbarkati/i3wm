import os, glob
import cv2
import numpy as np
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

X=[]
Y=np.zeros(52,dtype='int32')
i=0
for infile in glob.glob( os.path.join('shubh/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x=np.array(im)
    X.append(list(x))
    Y[i]=int(0)   
    i=i+1
for infile in glob.glob( os.path.join('abhi/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x=np.array(im)
    X.append(list(x))
    Y[i]=int(1)
    i=i+1

X=np.array(X)
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
model.compile(loss=keras.losses.mean_squared_logarithmic_error,optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=50)

im=cv2.imread('11.jpg')
x=[]
x.append(np.array(im))
x=np.array(x)
x=x/255
y=model.predict(x)
print y
model.save('my_face.h5')
