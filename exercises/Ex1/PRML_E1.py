# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:10:44 2017

@author: Jingyi
"""

""" Load CSV file """
import numpy as np
f = np.loadtxt('D:\Hanke\Python\PRML\locationData.csv')
print np.shape(f)
# print f.shape




""" Plot contents of matrix """
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.plot(f[:, 0], f[:, 1])

# ax = plt.subplot(1, 1, 1, projection='3d')
ax = Axes3D(plt.figure())
plt.plot(f[:, 0], f[:, 1], f[:, 2])
# projection='3d'无法识别，matplotlib版本问题




""" Basic image manipulation routines """
import matplotlib.image as im
img = im.imread('D:\Hanke\Python\PRML\oulu.jpg')
plt.imshow(img)
print np.shape(img)

print img.mean()
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
rgbmean = tuple((R.mean(), G.mean(), B.mean()))
print rgbmean

import scipy.ndimage.morphology as mor
white_img = mor.white_tophat(img, 10)
plt.figure()
plt.imshow(white_img)
# mean是什么意思




""" Load Matlab data """
from scipy.io import loadmat
with open('twoClassData.mat', 'rb') as f:
    mat = loadmat(f)
    print(mat.keys())
    X = mat["X"]
    y = mat["y"].ravel()

plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro')
plt.plot(X[y != 0, 0], X[y != 0, 1], 'bo')
# 有的点 y既等于0 又不等于0 如果先全部涂蓝再使y=0的涂红 红色会覆盖掉一些





""" Define a function """
# from numpy.linalg import inv
def normalize_data(X):
    X2 = X
    
    tmp0 = X[:, 0]
    tmp0mean = tmp0.mean()
    tmp0std = np.std(tmp0)
    tmp0 = tmp0 - tmp0mean
    tmp0 = tmp0 / tmp0std
    # tmp0 = list(tmp0)
    X2[:, 0] = tmp0
    
    tmp1 = X[:, 1]
    tmp1mean = tmp1.mean()
    tmp1std = np.std(tmp1)
    tmp1 = tmp1 - tmp1mean
    tmp1 = tmp1 / tmp1std
    # tmp1 = list(tmp1)
    X2[:, 1] = tmp1
    
    # X2 = np.vstack((tmp0, tmp1))
    # X3 = inv(X2)
    return X2
    

X_norm = normalize_data(X)
print np.mean(X_norm, axis = 0)
print np.std(X_norm, axis = 0)
# axis是什么


