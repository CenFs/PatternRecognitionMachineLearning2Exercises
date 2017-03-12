# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 21:44:43 2017

@author: Jingyi
"""

""" Implement gradient descent for log-loss """
import numpy as np
import matplotlib.pyplot as plt

def log_loss(w, X, y):
    L = 0
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(y[n] * np.dot(w, X[n])))
    return L
    
def grad(w, X, y):
    G = 0 # Accumulate gradient here.
    # Process each sample in X:
    for n in range(X.shape[0]):
        numerator = -y[n] * np.dot(X[n], np.exp(-y[n] * np.dot(w.T, X[n])))
        denominator = 1 + np.exp(-y[n] * np.dot(w.T, X[n]))
        G += numerator / denominator
    return G


if __name__ == "__main__":
    X = np.loadtxt('D:\Hanke\Python\PRML\log_loss_data\X.csv', skiprows=0, delimiter=',')
    y = np.loadtxt('D:\Hanke\Python\PRML\log_loss_data\y.csv', skiprows=0, dtype=int, delimiter=',')
    
    # print y[0]
    # print X[0].shape
    # w = np.zeros(X.shape)
    # w[0] = [0.9, 0.9]
    w = np.array([1, -1])
    step_size = 0.001
    # 4) Initialize empty lists for storing the path and accuracies:
    W = []
    accuracies = []
    
    for iteration in range(100):
        # 5) Apply the gradient descent rule.
        w = w - step_size * grad(w, X, y)
        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        
        # 7) Compute the accuracy (already done for you)
        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
        # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
        # Transform [0,1] coding to [-1,1] coding
        y_pred = 2 * y_pred - 1

        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        
        W.append(w)
    
    # 8) Below is a template for plotting. Feel free to rewrite if you prefer different style.
    
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






""" Select appropriate hyperparameters for the GTSRB data """
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
import glob

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
            hist = np.histogram(lbp, bins = n_bins)[0]
            X.append(hist)
            y.append(i)

X = np.array(X)
y = np.array(y)


X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf_list = [LogisticRegression(), SVC(kernel = 'linear')]
clf_name = ['LR', 'SVC']

C_range = 10.0 ** np.arange(-5, 1)

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(name, score)





""" Train ensemble methods with the GTSRB data """
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

'''
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('RandomForestClassifier', accuracy_score(y_test, y_pred))

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('ExtraTreesClassifier', accuracy_score(y_test, y_pred))

clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('AdaBoostClassifier', accuracy_score(y_test, y_pred))

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('GradientBoostingClassifier', accuracy_score(y_test, y_pred))

'''


ensemble_list = [(RandomForestClassifier(n_estimators=100), 'Random Forest'),
                 (ExtraTreesClassifier(n_estimators=100), 'Extra-Trees'),
                 (AdaBoostClassifier(n_estimators=100), 'AdaBoost'),
                 (GradientBoostingClassifier(n_estimators=100), 'GB-Trees')]

for ensemble, name in ensemble_list:
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(name, score)

