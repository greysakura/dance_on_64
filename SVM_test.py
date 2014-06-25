__author__ = 'LIMU_North'

import cv2
import numpy as np
import os
import math
from time import clock
import scikits
from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
linear_clf = svm.SVC(kernel='linear')
linear_clf.fit(X, y)

ggg = np.array([[0.0,0.0], [-1.0,-1.5]])
print ggg.shape

print linear_clf.decision_function(ggg)