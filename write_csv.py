__author__ = 'LIMU_North'

import cv2
import sys
import os
import math
import csv
import numpy as np
import gc
import re
import cPickle


# with open('C:/Cassandra/ground02/all_souls_000000_des.csv', 'r') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in spamreader:
#         print len(row)
#         print np.array(map(int,row))
#         print type(np.array(map(int,row)))

bbb = [1,2,3,4,5,6]
bbb = np.array(bbb)
aaa = [0,2,4,1,0,0]
aaa = np.array(aaa)
print aaa
print bbb
# aaa[0,:] = (np.int32(bbb > 3))
# print np.multiply(aaa,aaa)
#
# # true 0 or false -1
# str1 = 'what the fuxk'
# str2 = 'whats'
# print str1.find(str2)
# ccc = np.ones((1,15), int)
# print ccc.shape
# print aaa.shape
# print aaa.sum()
# print ccc.sum(axis = 1)[0]

aaa = [0,4]
aaa = np.array(aaa)
bbb = np.zeros((2,6), np.int32)
bbb[:,0] = (np.int32(aaa > 3).transpose())
print type(bbb[:,0])
ccc = np.reshape(np.int32(aaa > 3), [-1,1])
ccc = np.ones((2,6),np.int32)

ddd = np.zeros((111,222), np.int32)
# print bbb[:,0]
print ccc
# a = np.array([[1, 2, 3], [4, 5, -2],[6, 7, 9]])
# print a.shape
# print a[:,1].shape
# print (np.int32(a[:,1]>0).flatten()).shape

# f_write_go_on = open('C:/Cassandra/python_oxford/ccc.pkl','ab')
# cPickle.dump(ccc,f_write)
# cPickle.dump(ccc,open('C:/Cassandra/python_oxford/ccc.pkl','w'))

# f_read = open('C:/Cassandra/python_oxford/des_mat.pkl','rb')
# ccc = cPickle.load(f_read)
# f_read.close()

np.save('C:/Cassandra/python_oxford/ccc.npy',ddd)
ggg = np.load('C:/Cassandra/python_oxford/ccc.npy')

print ggg
# aaa = cPickle.load(f_read)
# print ccc
ddd = np.array([[0,8,0],[0,0,0],[7,8,1]])
single01 = np.array([0,0,0,1,0,1,0,0,0,0])
single02 = np.array([1,1,0,0,0,1,0,1,1,0])
xxx = ddd[0,np.where(ddd[0,:] > 1)]
print xxx.shape
print np.reshape(np.where(ddd[1,:] > 3)[0],(1,-1))
img01 = np.where(single01 == 1)[0]
img02 = np.where(single02 == 1)[0]
print np.union1d(img01,img02).shape
ggg = np.zeros((0,), np.int32)
print ggg.shape

filefile = open('C:/Cassandra/python_oxford/database_VW/all_souls_000000_VW.txt', 'r')
line = filefile.readline()
line = filefile.readline()
line = filefile.readline()
line = filefile.readline()
what = np.int32(line.split(','))
print np.int32(what >0)
print np.where((what>0) == 1)

filefile.close()

print  np.min((1,4))