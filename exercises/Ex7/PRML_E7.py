# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:43:01 2017

@author: Jingyi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, GridSearchCV

""" L2 penalized log-loss minimizer """
def log_loss(w, X, y):
    L = 0
    q = 1
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(-y[n] * np.dot(w, X[n])))
    L += q * np.dot(w.T, w)
    return L
    
def grad(w, X, y):
    G = 0
    q = 1
    for n in range(X.shape[0]):
        numerator = -y[n] * np.dot(X[n], np.exp(-y[n] * np.dot(w.T, X[n])))
        denominator = 1 + np.exp(-y[n] * np.dot(w.T, X[n]))
        G += numerator / denominator
    G += 2 * q * w
    return G
    

if __name__ == "__main__":
    X = np.loadtxt('D:\Hanke\Python\PRML\log_loss_data\X.csv', skiprows=0, delimiter=',')
    y = np.loadtxt('D:\Hanke\Python\PRML\log_loss_data\y.csv', skiprows=0, dtype=int, delimiter=',')
    
    w = np.array([1, -1])
    step_size = 0.001
    W = []
    accuracies = []
    
    for iteration in range(100):
        w = w - step_size * grad(w, X, y)
        print ("Iteration %d: w = %s (log-loss = %.2f)" % (iteration, str(w), log_loss(w, X, y)))
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
        y_pred = (y_prob > 0.5).astype(int)
        y_pred = 2 * y_pred - 1
        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        W.append(w)
    W = np.array(W)
    
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
    




""" recursive feature elimination approach """
data = loadmat('D:/Hanke/Python/PRML/arcene.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
y_train = y_train.ravel()
y_test  = y_test.ravel()

rfe = RFECV(estimator=LogisticRegression(), step=50, verbose=1, cv=5)
rfe.fit(X_train, y_train)
scores = rfe.grid_scores_
mask = rfe.support_
print mask # 全是true??
print rfe.n_features_

plt.plot(range(0,10001,50), rfe.grid_scores_)

y_pred = rfe.predict(X_test[:, mask])
print accuracy_score(y_test, y_pred)

"""
lr = LogisticRegression()
lr.fit(X_train[:, mask], y_train)
score = accuracy_score(y_test, lr.predict(X_train[:, mask]))
print score
"""


""" L1 penalized Logistic Regression for feature selection """
lr = LogisticRegression(penalty='l1')
# C_range = 10.0 ** np.arange(-5, 5)
C_range = np.logspace(0, 5, 20)
parameters = {'C': C_range}
gs = GridSearchCV(estimator=lr, param_grid=parameters, cv=10)
gs.fit(X_train, y_train)

clf = LogisticRegression(penalty='l1', C=gs.best_params_['C'])
clf.fit(X_train, y_train)
coef = clf.coef_
print np.count_nonzero(coef)

y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
