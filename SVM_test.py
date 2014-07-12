__author__ = 'LIMU_North'

import cv2
import numpy as np
import os
import math
from time import clock
import scikits
from sklearn import svm
import matplotlib.pyplot as pl


# we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 1001
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)

ratio = np.float64(n_samples_1)/n_samples_2
wclf = svm.SVC(kernel='linear', C = 1, class_weight={1: ratio})
print len(X)
wclf.fit(X, y)











# X = np.array([[0, 0], [1, 1]], np.int32)
# Z = np.array(X)
# print Z
# y = [0, 1]
# linear_clf = svm.SVC(kernel='linear')
# linear_clf.fit(X, y)
#
# ggg = np.array([[0.0,0.0], [-1.0,-1.5]])
# print ggg.shape
#
# print linear_clf.decision_function(ggg)
# print type(linear_clf.decision_function(ggg))
# print type(X)
#
# print [1]*5+[2]
#
# asdf = []
# for i in range(10):
#     asdf.append([i])
#
# print asdf
# print asdf[-5:]
#############################


# N = 5
# menMeans   = (20, 35, 30, 35, 27)
# # womenMeans = (25, 32, 34, 20, 25)
# menStd     = (2, 3, 4, 1, 2)
# # womenStd   = (3, 5, 2, 3, 3)
# ind = np.arange(N)    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence
#
# p1 = pl.bar(ind, menMeans,   width, color='r')
# # p2 = pl.bar(ind, womenMeans, width, color='y',
# #              bottom=menMeans, yerr=menStd)
#
# pl.ylabel('Scores')
# pl.title('Scores by group and gender')
# pl.xticks(ind+width/2., ('freq1', 'freq2', 'freq3', 'freq4', 'freq5') )
# pl.yticks(np.arange(0,81,10))
# # pl.legend( (p1[0]), ('Men') )
#
# pl.show()