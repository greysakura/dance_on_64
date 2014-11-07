__author__ = 'LIMU_North'
#-*- coding:utf-8 -*-
import math
import numpy as np
import subprocess
import logging
import os

# work_dir = 'C:/Users/LIMU_North/Downloads/svm_rank_windows/'
#
# train_returnCode  = subprocess.call(work_dir + 'svm_rank_learn.exe' + ' -c 3 '+ work_dir + 'train.dat ' + work_dir + 'model.dat')
#
# print train_returnCode
#
# test_returnCode = subprocess.call(work_dir + 'svm_rank_classify.exe' + ' '+ work_dir + 'train.dat ' + work_dir + 'model.dat ' +work_dir + 'prediction.dat')
#
# print test_returnCode

logging.basicConfig(filename = os.path.join(os.getcwd(), 'testlog.txt'),  level = logging.DEBUG, filemode = 'w', format = '%(asctime)s - %(levelname)s: %(message)s')
logging.info('Jackdaws love my big sphinx of quartz.')
AAA = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'),  (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j')]

AAA.append((10,'z'))
# print filter(lambda a: a[0] > 3, AAA)



# 定义一个Handler打印INFO及以上级别的日志到sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 设置日志打印格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# 将定义好的console日志handler添加到root logger
logging.getLogger('').addHandler(console)

logging.info('Jackdaws love my big sphinx of quartz.')

logger1 = logging.getLogger('myapp.area1')
logger2 = logging.getLogger('myapp.area2')

logger1.debug('Quick zephyrs blow, vexing daft Jim.')
logger1.info('How quickly daft jumping zebras vex.')
logger2.warning('Jail zesty vixen who grabbed pay from quack.')
logger2.error('The five boxing wizards jump quickly.')
logger1.info('how %d', 10)

logging.info('---------------- test over ------------------')
logging.shutdown()

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