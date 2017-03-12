# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:42:56 2017

@author: Jingyi
"""

""" Extract Local Binary Pattern features for classification """
import numpy as np
from skimage.feature import local_binary_pattern
import glob
import matplotlib.pyplot as plt

X = []
y = []
radius = 5
n_points = 8 # 8 * radius
n_bins = 256 # grayscale value 0-255

folders = sorted(glob.glob('D:\Hanke\Python\PRML\GTSRB_subset\*'))

for i,folder in enumerate(folders):
        files = glob.glob(folder + '/*')
        for f in files:
            f = plt.imread(f)
            lbp = local_binary_pattern(f, n_points, radius)
            # hist, _ = np.histogram(lbp, normed = True, bins = n_bins, range = (0, n_bins))
            hist = np.histogram(lbp, bins = n_bins)[0]
            X.append(hist)
            y.append(i)

X = np.array(X)
y = np.array(y)
print X.shape
print y.shape



""" Train classifiers for the GTSRB task """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
knn_y = clf.predict(X_test)
print 'KNN:'
print accuracy_score(y_test, knn_y)

lda = LDA()
lda.fit(X_train, y_train)
lda_y = lda.predict(X_test)
print 'LDA:'
print accuracy_score(y_test, lda_y)

svc = SVC()
svc.fit(X_train, y_train)
svc_y = svc.predict(X_test)
print 'SVC:'
print accuracy_score(y_test, svc_y)




