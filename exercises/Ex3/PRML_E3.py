# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:42:56 2017

@author: Jingyi
"""

""" sinusoid detector """
import numpy as np
import matplotlib.pyplot as plt

frequency = 0.1
n = np.arange(100)

s1 = np.zeros((500,), dtype = np.int)
s3 = np.zeros((300,), dtype = np.int)
s2 = np.cos(2 * np.pi * frequency * n)
y = np.concatenate((s1, s2, s3), axis = 0)

plt.figure()
f, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(y)

variance = 0.5
y_n = y + np.sqrt(variance) * np.random.randn(y.size)
ax2.plot(y_n)

# h = np.exp(-2 * np.pi * 1j * frequency * n)
d = np.convolve(s2, y_n, 'same')
ax3.plot(d)
# s2是原信号，现在已知原信号，所以用原信号和noisy信号卷积





""" different frequency and detector is random signal version """
frequency = 0.03
n = np.arange(100)

s1 = np.zeros((500,), dtype = np.int)
s3 = np.zeros((300,), dtype = np.int)
s2 = np.sin(2 * np.pi * frequency * n)
y = np.concatenate((s1, s2, s3), axis = 0)
plt.figure()
f, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(y)

variance = 0.5
y_n = y + np.sqrt(variance) * np.random.randn(y.size)
ax2.plot(y_n)

h = np.exp(-2 * np.pi * 1j * frequency * n)
d = np.abs(np.convolve(h, y_n, 'same'))
ax3.plot(d)
# 问题： 为什么要用绝对值表示random signal version? 因为h里有复数i
# 这题和上一题区别是什么? 区别在于这里不知道原始signal，只能用h来和noisy信号卷积
# 既然是random signal，n是怎么确定的?





""" Load a dataset of images split to training and testing """
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())
# ['images', 'data', 'target_names', 'DESCR', 'target']

import matplotlib.pyplot as plt
plt.gray()
plt.imshow(digits.images[0])
plt.show()
print digits.images[0]
print digits.data[0]
print digits.target[0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.20)





""" Train a classifier using the image data """
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")
clf.fit(X_train, y_train)
clf_y = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, clf_y)

