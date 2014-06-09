__author__ = 'LIMU_North'

from sklearn import svm
import numpy as np
X = [[0, 0], [1, 1], [0,1], [1,0]]
y = [1, 1, 0 ,0]
clf = svm.SVC()
print clf.fit(X, y)
print clf.predict([[2., 2.]])
print type(y)
print clf.support_
print clf.n_support_
# MA = np.array([[1,2],[2,3]])
# MB = np.array([[1,3],[-4,3]])
#
# ggg = np.array([[1],[2],[3],[4],[4],[5],[5]])
# ccc = ggg.repeat(repeats = 2, axis = 1)
#
#
# print MA
# print MB
# print MA*MB
# print ggg
# print ccc