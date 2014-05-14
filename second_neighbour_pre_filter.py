__author__ = 'LIMU_North'

import cv2
import numpy as np
import sys
import scipy
from scipy import spatial
from time import clock


img1 = cv2.imread('C:/Cassandra/graf_img1.png')
img2 = cv2.imread('C:/Cassandra/graf_img2.png')

sift = cv2.SIFT(nfeatures = 1000, edgeThreshold=0.01)
kpts1, desc1 = sift.detectAndCompute(img1,None)
kpts2, desc2 = sift.detectAndCompute(img2,None)
print len(kpts1)
print len(kpts2)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## second neighbour pre-filtering
nnThreshold = 0.8
time1 = clock()
tree = spatial.KDTree(desc1,leafsize = 10)
time2 = clock()
print 'time: ', time2-time1
query_desc = desc2
matcher = cv2.BFMatcher(cv2.NORM_L2)
time3 = clock()
raw_matches = matcher.knnMatch(desc2, trainDescriptors = desc1, k = 2)
time4 = clock()
print 'query time: ', time4-time3
print len(raw_matches)
print raw_matches[1][0].queryIdx
print raw_matches[1][0].trainIdx
print raw_matches[0][0].imgIdx
print raw_matches[1][0].imgIdx
print raw_matches[1][0].distance
print type(raw_matches)
good_match = []

for i in range(len(raw_matches)):
    if (raw_matches[i][0].distance / raw_matches[i][1].distance) <= float(pow(nnThreshold,2)):
        good_match.append(raw_matches[i])

print len(good_match)


# time3 = clock()
# query_answer =  tree.query(query_desc,k=2,p=2)
# time4 = clock()
# print 'query time: ', time4-time3
# print len(kpts1)
# print len(kpts2)
# print query_answer
# # print query_answer[0].shape


