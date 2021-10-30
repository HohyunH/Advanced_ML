# -*- coding: utf-8 -*-
# DO NOT CHANGE
import numpy as np
from itertools import product
from sklearn.svm import OneClassSVM
from scipy.sparse.csgraph import connected_components
import pandas as pd
import matplotlib.pyplot as plt


def get_adj_mat(X, svdd, num_cut):
    # X: n*p input matrix
    # svdd: trained svdd model by sci-kit learn using X
    # num_cut: number of cutting points on line segment
    #######OUTPUT########
    # return adjacent matrix size of n*n (if two points are connected A_ij=1)

    svdd.fit(X)
    support_vector = svdd.support_[svdd.dual_coef_[0] != 1]

    matrix = np.zeros(shape=(len(X), len(X)))
    coordinate = list(product(range(len(X)), range(len(X))))

    for row, col in coordinate:
        if row == col:
            matrix[row][col] = 0

        else:
            if (svdd.decision_function(np.linspace(X[row], X[col], num_cut)) >= np.min(svdd.decision_function(X[support_vector]))).sum() == num_cut:
                matrix[row][col] = 1
            else:
                matrix[row][col] = 0

    return matrix


def cluster_label(A, bsv):
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components

    labels = np.arange(len(A))
    matrix = A[np.setdiff1d(np.arange(len(A)), bsv), :][:, np.setdiff1d(np.arange(len(A)), bsv)]

    n, m = connected_components(matrix)

    for i in range(len(labels)):
        if i in bsv:
            labels[i] = -1
        else:
            labels[i] = m[0]
            m = np.delete(m, 0)

    return labels

#%%
ring = pd.read_csv('https://drive.google.com/uc?export=download&id=1_ygiOJ-xEPVSIvj3OzYrXtYc0Gw_Wa3a')
num_cut = 20
svdd = OneClassSVM(gamma=1, nu=0.2)

#%%

X = ring.values
ga_matrix = get_adj_mat(X, svdd, num_cut)

bsv = svdd.support_[svdd.dual_coef_[0] == 1]
unbound_sv = svdd.support_[svdd.dual_coef_[0] != 1]
non_sv = np.setdiff1d(np.arange(len(ring)), np.append(bsv, unbound_sv))

clus_label = cluster_label(ga_matrix, bsv)


tmp = []

for row, col in list(product(unbound_sv, non_sv)):
    if ga_matrix[row, col] == 1:
        tmp.append((row, col))
        
##########Plot1###################
# Get SVG figure (draw line between two connected points with scatter plots)
# draw decision boundary
# mark differently for nsv, bsv, and free sv

xmin, xmax = ring['X1'].min() - 0.5, ring['X1'].max() + 0.5
ymin, ymax = ring['X2'].min() - 0.5, ring['X2'].max() + 0.5

xx, yy, = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
zz = np.c_[xx.ravel(), yy.ravel()]
zz_pred = svdd.decision_function(zz)
plt.contour(xx, yy, zz_pred.reshape(xx.shape), levels=[0], linewidth=0.5, colors='k')
plt.scatter(ring['X1'][bsv], ring['X2'][bsv], marker='x', s=30, color='blue')
plt.scatter(ring['X1'][non_sv], ring['X2'][non_sv], marker='o', s=30, color='black')
plt.scatter(ring['X1'][unbound_sv], ring['X2'][unbound_sv], marker='o', facecolors='none', s=30, color='red')


for c in np.unique(unbound_sv):  ## drawing line
    x_lim = []
    y_lim = []

    for row, col in tmp:
        if row == c:
            x_lim.append(X[col][0])
            y_lim.append(X[col][1])
    x_lim.insert(0, X[c][0])
    y_lim.insert(0, X[c][1])
    plt.plot(x_lim, y_lim, color='black')

#%%
##########Plot2###################
# Clsuter labeling result
# different clusters should be colored using different color
# outliers (bounded support vectors) are marked with 'x'

xmin, xmax = ring['X1'].min() - 0.5, ring['X1'].max() + 0.5
ymin, ymax = ring['X2'].min() - 0.5, ring['X2'].max() + 0.5

xx, yy, = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
zz = np.c_[xx.ravel(), yy.ravel()]
zz_pred = svdd.decision_function(zz)
plt.contour(xx, yy, zz_pred.reshape(xx.shape), levels=[0], linewidth=1, colors='gray')
plt.scatter(ring['X1'][bsv], ring['X2'][bsv], marker='x', s=20, color='blue')
plt.scatter(X[clus_label == 0, 0], X[clus_label == 0, 1], marker='o', s=20, color='purple')
plt.scatter(X[clus_label == 1, 0], X[clus_label == 1, 1], marker='o', s=20, color='skyblue')
plt.scatter(X[clus_label == 2, 0], X[clus_label == 2, 1], marker='o', s=20, color='green')
plt.scatter(X[clus_label == 3, 0], X[clus_label == 3, 1], marker='o', s=20, color='yellow')