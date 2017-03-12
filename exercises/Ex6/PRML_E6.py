# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:41:04 2017

@author: Jingyi
"""

""" Load Traffic sign data for deep neural network processing """
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten


def MaxMinNormalization(x, Max, Min):
    Max *= 1.0
    Min *= 1.0
    return (x - Min) / (Max - Min)

X = []
y = []

folders = sorted(glob.glob('D:\Hanke\Python\PRML\GTSRB_subset_2\*'))

for i,folder in enumerate(folders):
        files = glob.glob(folder + '/*')
        for f in files:
            f = np.transpose(plt.imread(f))
            f = MaxMinNormalization(f, np.max(f), np.min(f))
            X.append(f)
            y.append(i)

X = np.array(X)
y = np.array(y)


y = np_utils.to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)




""" Define the network in Keras """

N = 10 # Number of feature maps
w, h = 3, 3 # Conv. window size
model = Sequential()
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        activation = 'relu',
                        input_shape = (3,64,64)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(2, activation = 'sigmoid'))



""" Compile and train the net """
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, nb_epoch=20, validation_data = [X_test, y_test])
