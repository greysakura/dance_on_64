__author__ = 'LIMU_North'
#-*- coding:utf-8 -*-
import math
import numpy as np
import subprocess
import logging
import os
import cv2
top_dir = 'C:/Cassandra/python_oxford/'
query_goto_dir = 'C:/Cassandra/query_object/'
DQE_reranking_dir = top_dir + 'DQE_reranking/'
Ranking_SVM_reranking_dir = top_dir + 'Ranking_SVM_reranking/'
dirVisualization = top_dir + 'Visual/'
top_retrieval_num = 5062

# work_dir = 'C:/Users/LIMU_North/Downloads/svm_rank_windows/'
#
# train_returnCode  = subprocess.call(work_dir + 'svm_rank_learn.exe' + ' -c 3 '+ work_dir + 'train.dat ' + work_dir + 'model.dat')
#
# print train_returnCode
#
# test_returnCode = subprocess.call(work_dir + 'svm_rank_classify.exe' + ' '+ work_dir + 'train.dat ' + work_dir + 'model.dat ' +work_dir + 'prediction.dat')
#
# print test_returnCode

# logging.basicConfig(filename = os.path.join(os.getcwd(), 'testlog.txt'),  level = logging.DEBUG, filemode = 'w', format = '%(asctime)s - %(levelname)s: %(message)s')
# logging.info('Jackdaws love my big sphinx of quartz.')
# AAA = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'),  (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j')]
#
# AAA.append((10,'z'))
# # print filter(lambda a: a[0] > 3, AAA)
#
#
#
# # 定义一个Handler打印INFO及以上级别的日志到sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # 设置日志打印格式
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# # 将定义好的console日志handler添加到root logger
# logging.getLogger('').addHandler(console)
#
# logging.info('Jackdaws love my big sphinx of quartz.')
#
# logger1 = logging.getLogger('myapp.area1')
# logger2 = logging.getLogger('myapp.area2')
#
# logger1.debug('Quick zephyrs blow, vexing daft Jim.')
# logger1.info('How quickly daft jumping zebras vex.')
# logger2.warning('Jail zesty vixen who grabbed pay from quack.')
# logger2.error('The five boxing wizards jump quickly.')
# logger1.info('how %d', 10)
#
# logging.info('---------------- test over ------------------')
# logging.shutdown()

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

# A = np.array([[0,-1,1,-1],[1,0,-1,1],[-1,1,0,-1],[1,-1,1,0]])
# print np.linalg.det(A)

#############

# myfile = open('C:/Users/LIMU_North/Dropbox/Liu Box/ForLatex/mAP_FCV.csv', 'r')
#
# myfile.readline()
#
# tmplst = []
#
# for line in myfile:
#     tmplst.append(line.split('\n')[0].split(','))
#
# mymat = np.array(tmplst, np.float32)
# myfile.close()
# print mymat
#
# newMat = np.zeros([11,5], np.float32)
# print mymat[0:5,1]
# print np.average(mymat[0:5,1])
#
# for i in range(11):
#     for j in range(5):
#         newMat[i,j] = np.average(mymat[(5*i):(5*i + 5),j+1])
#
# print 'newMat:'
# print newMat
# newFile = open('C:/Users/LIMU_North/Dropbox/Liu Box/ForLatex/mAP_Average.csv', 'w')
# for i in range(newMat.shape[0]):
#     for j in range(newMat.shape[1]):
#         newFile.write(str(newMat[i,j]))
#         if j < (newMat.shape[1]-1):
#             newFile.write(',')
#     newFile.write('\n')
#
# newFile.close()

##################################

# mat_tmp = cv2.imread('C:/Cassandra/python_oxford/Visual/SV_reranking/balliol_5_query/query_15.jpg')
# gray = cv2.cvtColor(mat_tmp,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img', mat_tmp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# sift = cv2.SIFT()
# kp, des = sift.detectAndCompute(gray,None)
# img=cv2.drawKeypoints(mat_tmp,kp, flags = 4, color = (0,0,255))
#
# cv2.imwrite('C:/Cassandra/python_oxford/Visual/SV_reranking/balliol_5_query/query_15_SIFT.jpg', img)
# cv2.imshow('new img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##################################
mat_query = cv2.imread('C:/Cassandra/python_oxford/Visual/SV_reranking/balliol_5_query/query_15.jpg')
mat_target = cv2.imread('C:/Cassandra/python_oxford/Visual/SV_reranking/balliol_5_query/result_1.jpg')


print mat_query.shape
print type(mat_query[0][0,0])
#### decide the size of new image.
new_height = max(mat_query.shape[0], mat_target.shape[0])
new_length = mat_query.shape[1] + mat_target.shape[1]

cv2.rectangle(mat_query,(0,0),(mat_query.shape[1], mat_query.shape[0]), color = (0,0,255), thickness=5)
mat_bg = np.ones([new_height,new_length,3], np.uint8)*255

# np.copyto(mat_query, mat_black, [mat_query.shape[0],mat_query.shape[1],3])

## copy two image into black image.
mat_bg[0:mat_query.shape[0], 0:mat_query.shape[1],:] = np.copy(mat_query)
mat_bg[0:mat_target.shape[0], mat_query.shape[1]:(mat_query.shape[1]+mat_target.shape[1]),:] = np.copy(mat_target)
print mat_query.shape
print mat_bg[0:mat_query.shape[0], 0:mat_query.shape[1],:].shape

## rectangle
cv2.rectangle(mat_bg,(mat_query.shape[1] + 430,36),(mat_query.shape[1] + 880, 688), color = (0,0,255), thickness=5)
##
cv2.line(mat_bg, (0,0), (mat_query.shape[1] + 430,36), color = (0,0,255), thickness=3, lineType= 8)
cv2.line(mat_bg, (0,mat_query.shape[0]), (mat_query.shape[1] + 430,688), color = (0,0,255), thickness=3, lineType= 8)
cv2.line(mat_bg, (mat_query.shape[1],0), (mat_query.shape[1] + 880,36), color = (0,0,255), thickness=3, lineType= 8)
cv2.line(mat_bg, (mat_query.shape[1],mat_query.shape[0]), (mat_query.shape[1] + 880, 688), color = (0,0,255), thickness=3, lineType= 8)
cv2.imshow('img', mat_query)
cv2.imshow('blkimg', mat_bg)
cv2.imwrite('C:/Cassandra/python_oxford/Visual/SV_reranking/balliol_5_query/query_15_combine.jpg',mat_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()