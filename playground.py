__author__ = 'LIMU_North'

import math
import numpy as np

A = np.array([[1,1],[1,1],[0,1]], float)

part01 = np.linalg.inv(np.dot(A.transpose(),A))
print part01
print np.dot(A.transpose(),A)
print A
b = np.array([[4],[4],[6]],float)
print b

part02 = np.dot(A.transpose(), b)
print A.dot(part01).dot(part02)

print math.pow(2,64)