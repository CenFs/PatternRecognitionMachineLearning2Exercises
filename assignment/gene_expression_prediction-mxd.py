#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gene Expression Prediction

@author: mingxiaodong
"""
import csv
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

gene_x_train = np.loadtxt('x_train.csv', skiprows=1, dtype=int, delimiter=',')
gene_x_train_data_list = []
gene_x_train_data_list_mean = []
gene_x_train_data_list_std = []

gene_y_train = np.loadtxt('y_train.csv', skiprows=1, dtype=int, delimiter=',')
gene_y_train_data_list = []

gene_x_test = np.loadtxt('x_test.csv', skiprows=1, dtype=int, delimiter=',')
gene_x_test_data_list = []
gene_x_test_data_list_mean = []
gene_x_test_data_list_std = []

for i in range(len(gene_x_train)):
    if (i % 100) == 0:
        gene_x_train_data_list.append(gene_x_train[i:(i + 100), 1:])
        gene_x_train_data_list_mean.append(np.mean(gene_x_train[i:(i + 100), 1:], axis=0))
        gene_x_train_data_list_std.append(np.std(gene_x_train[i:(i + 100), 1:], axis=0))

gene_y_train_data_list = gene_y_train[:, 1]

for i in range(len(gene_x_test)):
    if (i % 100) == 0:
        gene_x_test_data_list.append(gene_x_test[i:(i + 100), 1:])
        gene_x_test_data_list_mean.append(np.mean(gene_x_test[i:(i + 100), 1:], axis=0))
        gene_x_test_data_list_std.append(np.mean(gene_x_test[i:(i + 100), 1:], axis=0))

#clf = SVC(probability=True)
#clf.fit(gene_x_train_data_list_std, gene_y_train_data_list)
#gene_y_test_data_list = clf.predict(gene_x_test_data_list_std)
#gene_y_test_data_list_proba = clf.predict_proba(gene_x_test_data_list_std)

clf = LogisticRegression()
clf.fit(gene_x_train_data_list_std, gene_y_train_data_list)
gene_y_test_data_list = clf.predict(gene_x_test_data_list_std)
gene_y_test_data_list_proba = clf.predict_proba(gene_x_test_data_list_std)

gene_y_test_data_list_id = []

with open("predict4.csv", 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ["GeneId", "Prediction"], delimiter = ',')
    writer.writeheader()
    
for i in range(len(gene_y_test_data_list)):
    gene_y_test_data_list_id.append([i+1, gene_y_test_data_list_proba[i][1]])

np.savetxt("predict4.csv", gene_y_test_data_list_id, delimiter=",")