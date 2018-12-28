############################################
#                                          #
# CSE4088-Introduction to Machine Learning #
#                 Project                  #
#         Hilal EKİNCİ    - 150114057      #
#        Oğuzhan BÖLÜKBAŞ - 150114022      #
#                                          #
############################################

import cv2
import numpy as np
from random import shuffle
import tqdm
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time

train_data = 'kokturk_train'
test_data = 'kokturk_test'
def one_hot_label(img):

    label = img.split('.')[0]
    if label == 'zero':
        ohl = np.array([1,0,0,0,0,0,0,0,0,0])
    elif label == 'one':
        ohl = np.array([0,1,0,0,0,0,0,0,0,0])
    elif label == 'two':
        ohl = np.array([0,0,1,0,0,0,0,0,0,0])
    elif label == 'three':
        ohl = np.array([0,0,0,1,0,0,0,0,0,0])
    elif label == 'four':
        ohl = np.array([0,0,0,0,1,0,0,0,0,0])
    elif label == 'five':
        ohl = np.array([0,0,0,0,0,1,0,0,0,0])
    elif label == 'six':
        ohl = np.array([0,0,0,0,0,0,1,0,0,0])
    elif label == 'seven':
        ohl = np.array([0,0,0,0,0,0,0,1,0,0])
    elif label == 'eight':
        ohl = np.array([0,0,0,0,0,0,0,0,1,0])
    elif label == 'nine':
        ohl = np.array([0,0,0,0,0,0,0,0,0,1])

    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm.tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    print("\nTraining images:", len(train_images))
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm.tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        #test_images.append([np.array(img), one_hot_label(i)])
        test_images.append([np.array(img)])

    shuffle(test_images)
    print("Testing images: ", len(test_images))
    return test_images

timeStart= time.time()
from keras.models import  Sequential
from keras.layers import  *
from keras.optimizers import  *

training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,28,28,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,28,28,1)
#tst_lbl_data = np.array([i[1] for i in testing_images])

# Training Part
model = Sequential()

model.add(InputLayer(input_shape=[28,28,1]))
model.add(Conv2D(filters=32, kernel_size=5,strides=1,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=50, kernel_size=5,strides=1,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=80, kernel_size=5,strides=1,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr=1e-3)

model.compile(optimizer = optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x= tr_img_data, y= tr_lbl_data, epochs=100, batch_size=100)
model.summary()
timeFinal = time.time()
model.save('my_model.h5')

fig = plt.figure(figsize=(20,20))

for cnt, data in enumerate(testing_images): # Testing Part
    y = fig.add_subplot(10,20, cnt+1)
    img = data[0]
    data = img.reshape(1,28,28,1)
    model_out = model.predict([data])
    print(model_out)

    if np.argmax(model_out) == 0:
        str_label = '0'
    elif np.argmax(model_out) == 1:
        str_label = '1'
    elif np.argmax(model_out) == 2:
        str_label = '2'
    elif np.argmax(model_out) == 3:
        str_label = '3'
    elif np.argmax(model_out) == 4:
        str_label = '4'
    elif np.argmax(model_out) == 5:
        str_label = '5'
    elif np.argmax(model_out) == 6:
        str_label = '6'
    elif np.argmax(model_out) == 7:
        str_label = '7'
    elif np.argmax(model_out) == 8:
        str_label = '8'
    elif np.argmax(model_out) == 9:
        str_label = '9'


    y.imshow(img, cmap= 'gray')
    plt.title(str_label)
    print(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)


plt.show()
print("Time taken in seconds: ",timeFinal - timeStart)