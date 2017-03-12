#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 4.4 Extract Local Binary Pattern features for classification.
         4.5 Train classifiers for the GTSRB task.

@author: mingxiaodong
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern # 主要用于图像识别
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

X = [] # 存放直方图值
y = [] # 存放label

folders = sorted(glob.glob('GTSRB_subset/*')) # 设置文件读取路径

for i,folder in enumerate(folders):
        file_list = glob.glob(folder+'/*')
        for file in file_list:
            file = plt.imread(file)
            lbp = local_binary_pattern(file, 8, 3) # 以每个像素点为圆心，3为半径，取八个点与该像素比较，得到阈值
            hist = np.histogram(lbp, bins=256)[0] # 因为灰度值范围是0-255，所以bin取256，[0]返回直方图的值
            X.append(hist)
            y.append(i)

X = np.array(X)
y = np.array(y)
print(X.shape)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

clf = KNeighborsClassifier()
clf.fit(xtrain, ytrain)
predict_y = clf.predict(xtest)
print('KNN', accuracy_score(ytest, predict_y))

lda = LDA()
lda.fit(xtrain, ytrain)
predict_y = lda.predict(xtest)
print('LDA', accuracy_score(ytest, predict_y))

svc = SVC()
svc.fit(xtrain, ytrain)
predict_y = svc.predict(xtest)
print('SVC', accuracy_score(ytest, predict_y))