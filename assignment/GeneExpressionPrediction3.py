# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:41:47 2017

@author: Jingyi
"""

import numpy as np
import csv
import os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split


if __name__== '__main__':

    data_path = "D:/Hanke/Python/GeneExpressionPrediction/"  # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.
    
    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv", 
                         delimiter = ",", skiprows = 1)
    x_test  = np.loadtxt(data_path + os.sep + "x_test.csv", 
                         delimiter = ",", skiprows = 1)    
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv", 
                         delimiter = ",", skiprows = 1)
    
    print "All files loaded. Preprocessing..."

    # remove the first column(Id)
    x_train = x_train[:, 1:] 
    x_test  = x_test[:, 1:]   
    y_train = y_train[:, 1:] 
    
    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test  = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test  = np.split(x_test, num_genes_test)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)
    """
    tmp = []
    for i in range(15485):
        tmp.append([np.array([1]), x_train])
    tmp = np.array(tmp)
    tmp2 = np.asarray(tmp)
    """
    
    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))
    
    print('Data preprocessing done...')
    print("-" * 30)
    
    
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
    
    print (x_train.shape[1], x_train.shape[2])
    
    print("CNN...")
    model = Sequential()
    model.add(Convolution2D(nb_filter = 64,
                            nb_col = 10,
                            nb_row = 1,
                            activation = 'relu',
                            subsample = (1,1),
                            input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Convolution2D(nb_filter = 64,
                            nb_col = 10,
                            nb_row = 1,
                            subsample = (1,1),
                            activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    
    print("Compiling...")
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    print("Fitting...")
    model.fit(x_train, y_train, nb_epoch=10, batch_size=32)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    # print accuracy_score(y_test, y_pred)
    print('\n\nCNN DONE!\n\n')
    
    y_test = []
    for i in range(len(y_pred)):
        y_test.append([i+1, y_pred_proba[i][0]])
    
    
    filename = "CNN-batch32-filter64-64-colrow10-1-relu-subsample1.csv"
    print("Writing " + filename + " ...")
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["GeneId", "Prediction"], delimiter=",")
        writer.writeheader()
    np.savetxt(filename, y_test, delimiter=",")
    print("Writing DONE.")
    print("\n\n\n------- ALL DONE!!! -------\n\n\n")

