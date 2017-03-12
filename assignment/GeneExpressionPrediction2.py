# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:22:15 2017

@author: Jingyi
"""

import numpy as np
import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


if __name__== '__main__':
    
    gene_x_train = np.loadtxt('x_train.csv', skiprows=1, dtype=int, delimiter=',')
    gene_x_train_data_list = []
    gene_x_train_data_list_mean = []
    gene_x_train_data_list_sum = []
    gene_x_train_data_list_std = []
    
    gene_y_train = np.loadtxt('y_train.csv', skiprows=1, dtype=int, delimiter=',')
    gene_y_train_data_list = []
    
    gene_x_test = np.loadtxt('x_test.csv', skiprows=1, dtype=int, delimiter=',')
    gene_x_test_data_list = []
    gene_x_test_data_list_mean = []
    gene_x_test_data_list_sum = []
    gene_x_test_data_list_std = []
    
    for i in range(len(gene_x_train)):
        if (i % 100) == 0:
            gene_x_train_data_list.append(gene_x_train[i:(i + 100), 1:])
            gene_x_train_data_list_mean.append(np.mean(gene_x_train[i:(i + 100), 1:], axis=0))
            gene_x_train_data_list_sum.append(np.sum(gene_x_train[i:(i + 100), 1:], axis=0))
            gene_x_train_data_list_std.append(np.std(gene_x_train[i:(i + 100), 1:], axis=0))
    
    gene_y_train_data_list = gene_y_train[:, 1]
    
    for i in range(len(gene_x_test)):
        if (i % 100) == 0:
            gene_x_test_data_list.append(gene_x_test[i:(i + 100), 1:])
            gene_x_test_data_list_mean.append(np.mean(gene_x_test[i:(i + 100), 1:], axis=0))
            gene_x_test_data_list_sum.append(np.sum(gene_x_test[i:(i + 100), 1:], axis=0))
            gene_x_test_data_list_std.append(np.std(gene_x_test[i:(i + 100), 1:], axis=0))
    
    
    """
    #clf = SVC(probability=True)
    #clf.fit(gene_x_train_data_list_std, gene_y_train_data_list)
    #gene_y_test_data_list = clf.predict(gene_x_test_data_list_std)
    #gene_y_test_data_list_proba = clf.predict_proba(gene_x_test_data_list_std)
    
    clf = LogisticRegression()
    clf.fit(gene_x_train_data_list_std, gene_y_train_data_list)
    gene_y_test_data_list = clf.predict(gene_x_test_data_list_std)
    gene_y_test_data_list_proba = clf.predict_proba(gene_x_test_data_list_std)
    """

    x_train = np.array(gene_x_train_data_list_mean)
    y_train = np.array(gene_y_train_data_list)
    x_test  = np.array(gene_x_test_data_list_mean)
    y_train = np.ravel(y_train)
    
    classifiers = [('KNN2', KNeighborsClassifier(n_neighbors=10)), 
                   ('LDA2', LDA()), 
                   ('SVC2', SVC(probability=True)),
                   ('LR2', LogisticRegression()),
                   ('RandomForest2', RandomForestClassifier(n_estimators=100)), # default: n_estimators=10
                   ('ExtraTrees2', ExtraTreesClassifier(n_estimators=100)), 
                   ('AdaBoost2', AdaBoostClassifier(n_estimators=100)), 
                   ('GradientBoosting2', GradientBoostingClassifier(n_estimators=100)),
                   # ('Dense2', Sequential()),
                   # ('CNN2', Sequential()),
                   ]
    for name, clf in classifiers:
        print(name + '...')
        if name == "Dense2":
            clf.add(Dense(1000, input_dim=500, init='uniform', activation='relu'))
            clf.add(Dense(500, init='uniform', activation='relu'))
            clf.add(Dense(1, init='uniform', activation='sigmoid'))
            print("Compiling...")
            clf.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            print("Fitting...")
            clf.fit(x_train, y_train, nb_epoch=20, batch_size=10)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)
            print('\n\n\n' + name + ' DONE!\n\n')
            y_test = []
            for i in range(len(y_pred)):
                y_test.append([i+1, y_pred_proba[i][0]])
        if name == "CNN2":
            clf.add(Convolution1D(nb_filter=1000, input_shape=(15485,500), border_mode='same'))
            clf.add(Convolution1D(nb_filter=500, border_mode='same'))
            clf.add(Dense(1, init='uniform', activation='sigmoid'))
            print("Compiling...")
            clf.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            print("Fitting...")
            clf.fit(x_train, y_train, nb_epoch=20, batch_size=10)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)
            print('\n\n\n' + name + ' DONE!\n\n')
            y_test = []
            for i in range(len(y_pred)):
                y_test.append([i+1, y_pred_proba[i][0]])
        if name != "CNN2" and name != "Dense2":
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)
            print('\n' + name + ' DONE!\n')
            y_test = []
            for i in range(len(y_pred)):
                y_test.append([i+1, y_pred_proba[i][1]])
            
        filename = name + ".csv"
        print("Writing " + filename + " ...")
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=["GeneId", "Prediction"], delimiter=",")
            writer.writeheader()
        np.savetxt(filename, y_test, delimiter=",")
        print("Writing DONE.")
        print("-" * 30)
    

    print("\n\n\n------- ALL DONE!!! -------\n\n\n")
    