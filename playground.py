__author__ = 'LIMU_North'

import math
import numpy as np
import subprocess

work_dir = 'C:/Users/LIMU_North/Downloads/svm_rank_windows/'

train_returnCode  = subprocess.call(work_dir + 'svm_rank_learn.exe' + ' -c 3 '+ work_dir + 'train.dat ' + work_dir + 'model.dat')

print train_returnCode

test_returnCode = subprocess.call(work_dir + 'svm_rank_classify.exe' + ' '+ work_dir + 'train.dat ' + work_dir + 'model.dat ' +work_dir + 'prediction.dat')

print test_returnCode



# A = np.array([[2,-1],[-1,2]], float)
#
# part01 = np.linalg.inv(A)
#
# print A
# print part01
#
# print type(A[0])
# print part01
# print np.dot(A.transpose(),A)
# print A
# b = np.array([[4],[4],[6]],float)
# print b
#
# part02 = np.dot(A.transpose(), b)
# print A.dot(part01).dot(part02)
#
# print math.pow(2,64)