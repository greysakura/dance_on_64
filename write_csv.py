__author__ = 'LIMU_North'

import cv2
import sys
import os
import math
import csv
import numpy as np
import gc
import re


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
print np.multiply(aaa,aaa)

# true 0 or false -1
str1 = 'what the fuxk'
str2 = 'whats'
print str1.find(str2)
