
# coding: utf-8

# In[50]:


import os, glob
import cv2

import numpy as np
import pandas
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
import keras
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras import regularizers


# In[ ]:


l=0
for infile in glob.glob( os.path.join('negatives/', "*.jpg") ):
    l=l+1
for infile in glob.glob( os.path.join('positives/', "*.jpg") ):
    l=l+1


# In[125]:


X=[]
Y=np.zeros(l,dtype='int32')
i=0
for infile in glob.glob( os.path.join('negatives/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY
    x=np.array(im)
    X.append(list(im))
    Y[i]=int(0)
    i=i+1
for infile in glob.glob( os.path.join('positives/', "*.jpg") ):
    im = cv2.imread(infile)
    #gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x=np.array(im)
    X.append(list(im))
    Y[i]=int(1)
    i=i+1

#X=X/255
X=np.array(X)
print X.shape

# In[127]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(20, 20), strides=(1, 1),
                 activation='relu',
                 input_shape=[50,50,3],kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(50, activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(1, activation='sigmoid',kernel_initializer='random_uniform'))
model.compile(loss=keras.losses.logcosh,
              optimizer='SGD',
              metrics=['accuracy'])
model.fit(X,Y,epochs=50)


# In[13]:


im=cv2.imread('predict.jpg')
x=[]
x.append(np.array(im))
x=np.array(x)
x=x/255
y=model.predict(x)
print y
model.save('my_face.h5')


# In[21]:


import os, glob
import cv2

ulpath = "./shubh_color/"
c=0
for infile in glob.glob( os.path.join(ulpath, "*.jpg") ):
    im = cv2.imread(infile)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2=cv2.resize(gray_image,(100,100))
    cv2.imwrite('./negatives/'+str(c)+'.jpg',im2)
    c=c+1


