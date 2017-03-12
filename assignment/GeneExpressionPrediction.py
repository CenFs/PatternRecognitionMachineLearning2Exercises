# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:50:16 2017

@author: Jingyi
"""

import numpy as np
import os
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

"""
OUTPUT:
GeneId, Prediction.
1-3871, likehood
"""

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

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test  = [g.ravel() for g in x_test]
    
    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_train = np.ravel(y_train)
    
    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.
    
    print("x_train shape is %s" % str(x_train.shape))    
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))
    
    print('Data preprocessing done...')
    print("-" * 30)

    """
    print("Next steps FOR YOU:")
    print("1. Define a classifier using sklearn")
    print("2. Assess its accuracy using cross-validation (optional)")
    print("3. Fine tune the parameters and return to 2 until happy (optional)")
    print("4. Create submission file. Should be similar to y_train.csv.")
    print("5. Submit at kaggle.com and sit back.")
    """

    """
    print('KNN...')
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    knn_y = clf.predict(x_test)
    knn_y_proba = clf.predict_proba(x_test)
    # print('KNN:', accuracy_score(y_test, knn_y))
    print('DONE!')
    print("-" * 30)
    print('LDA...')
    lda = LDA()
    lda.fit(x_train, y_train)
    lda_y = lda.predict(x_test)
    lda_y_proba = clf.predict_proba(x_test)
    # print('LDA:', accuracy_score(y_test, lda_y))
    print('DONE!')
    print("-" * 30)
    print('SVC...')
    svc = SVC()
    svc.fit(x_train, y_train)
    svc_y = svc.predict(x_test)
    svc_y_proba = clf.predict_proba(x_test)
    # print('SVC:', accuracy_score(y_test, svc_y))
    print('DONE!')
    print("-" * 30)
    print('RandomForest...')
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    randomForest_y = clf.predict(x_test)
    randomForest_y_proba = clf.predict_proba(x_test)
    # print('RandomForestClassifier', accuracy_score(y_test, y_pred))
    print('DONE!')
    print("-" * 30)
    print('ExtraTrees...')
    clf = ExtraTreesClassifier()
    clf.fit(x_train, y_train)
    extraTrees_y = clf.predict(x_test)
    extraTrees_y_proba = clf.predict_proba(x_test)
    # print('ExtraTreesClassifier', accuracy_score(y_test, y_pred))
    print('DONE!')
    print("-" * 30)
    print('AdaBoost...')
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    adaBoost_y = clf.predict(x_test)
    gradientBoosting_y_proba = clf.predict_proba(x_test)
    # print('AdaBoostClassifier', accuracy_score(y_test, y_pred))
    print('DONE!')
    print("-" * 30)
    print('GradientBoosting...')
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    gradientBoosting_y = clf.predict(x_test)
    gradientBoosting_y_proba = clf.predict_proba(x_test)
    # print('GradientBoostingClassifier', accuracy_score(y_test, y_pred))
    print('DONE!')
    print("-" * 30)
    
    # First we initialize the model. "Sequential" means there are no loops.
    clf = Sequential()
    # Add layers one at the time. Each with 100 nodes.
    clf.add(Dense(100, input_dim=2, activation = 'sigmoid'))
    clf.add(Dense(100, activation = 'sigmoid'))
    clf.add(Dense(1, activation = 'sigmoid'))
    # The code is compiled to CUDA or C++
    clf.compile(loss='mean_squared_error', optimizer='sgd')
    clf.fit(x_train, y_train, nb_epoch=20, batch_size=16) # takes a few seconds
    
    # CNN
    model = Sequential()
    model.add(Convolution2D(nb_filter = 10,
                            nb_col = 3,
                            nb_row = 3,
                            activation = 'relu',
                            input_shape = (3,64,64)))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(nb_filter = 10,
                            nb_col = 3,
                            nb_row = 3,
                            activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(2, activation = 'sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test, batch_size=32, verbose=1)
    """
    
    classifiers = [# ('KNN1', KNeighborsClassifier(n_neighbors=10)), 
                   # ('LDA1', LDA()), 
                   # ('SVC1', SVC(probability=True)),
                   # ('LR1', LogisticRegression()),
                   # ('RandomForest1', RandomForestClassifier(n_estimators=100)), # default: n_estimators=10
                   # ('ExtraTrees1', ExtraTreesClassifier(n_estimators=100)), 
                   # ('AdaBoost1', AdaBoostClassifier(n_estimators=100)), 
                   # ('GradientBoosting1', GradientBoostingClassifier(n_estimators=100)),
                   # ('Dense1', Sequential()),
                   ('CNN1', Sequential()),
                   ]
    for name, clf in classifiers:
        print(name + '...')
        if name == "Dense1":
            clf.add(Dense(1000, input_dim=500, init='uniform', activation='relu'))
            clf.add(Dense(500, init='uniform', activation='relu'))
            clf.add(Dense(1, init='uniform', activation='sigmoid'))
            print("Compiling...")
            clf.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            print("Fitting...")
            clf.fit(x_train, y_train, nb_epoch=10, batch_size=10)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)
            print('\n\n\n' + name + ' DONE!\n\n')
            y_test = []
            for i in range(len(y_pred)):
                y_test.append([i+1, y_pred_proba[i][0]])
        if name == "CNN1":
            clf.add(Convolution1D(nb_filter=64, input_shape=(15485,500), activation='relu', subsample=1))
            clf.add(Convolution1D(nb_filter=64, activation='relu', subsample=1))
            clf.add(Dense(1, init='uniform', activation='sigmoid'))
            print("Compiling...")
            clf.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            print("Fitting...")
            clf.fit(x_train, y_train, nb_epoch=10, batch_size=32)
            y_pred = clf.predict(x_test)
            y_pred_proba = clf.predict_proba(x_test)
            print('\n\n\n' + name + ' DONE!\n\n')
            y_test = []
            for i in range(len(y_pred)):
                y_test.append([i+1, y_pred_proba[i][0]])
        if name != "CNN1" and name != "Dense1":
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

"""
if __name__ == "__main__":
    lastGeneId = 3871
    rowsy = range(1, lastGeneId+1)
    rowsX = range(1, lastGeneId*100+1)
    # X_train(GeneId, H3K4me3, H3K4me1, H3K36me3, H3K9me3, H3K27me3)
    X_train = np.loadtxt('D:\Hanke\Python\GeneExpressionPrediction\X_train.csv', skiprows=1, dtype=int, delimiter=',')
    # X_test(GeneId, H3K4me3, H3K4me1, H3K36me3, H3K9me3, H3K27me3)
    X_test = np.loadtxt('D:\Hanke\Python\GeneExpressionPrediction\X_test.csv', skiprows=1, dtype=int, delimiter=',')
    # y_train(GeneId(1, 15485), Prediction(0, 1))
    y_train = np.loadtxt('D:\Hanke\Python\GeneExpressionPrediction\y_train.csv', skiprows=1, dtype=int, delimiter=',')
    # 每个X有100个train和25个test
    print X_train.shape
    print X_test.shape
    print y_train.shape
    with open('D:\Hanke\Python\GeneExpressionPrediction\y_train.csv', 'r') as fin:
        reader = csv.reader(fin)
        result = [[int(row[1])] for i,row in enumerate(reader) if i in rowsy]
    y_train = np.array(result)
    with open('D:\Hanke\Python\GeneExpressionPrediction\X_train.csv', 'r') as fin:
        reader = csv.reader(fin)
        result = [[int(s) for s in row] for i,row in enumerate(reader) if i in rowsX]
    X_train = np.array(result)
    
    
    #y_lable = y_train[:, 1]
    #for i in y_train.shape[0]:
        #for j in y_train.shape[0]:
            #if j < i * 100:
                #X[j] = X_train[j]
"""

