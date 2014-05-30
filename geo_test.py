__author__ = 'LIMU_North'

import cv2
import numpy as np
import sys
import scipy
from scipy import spatial
from time import clock

img01 = cv2.imread('C:/Cassandra/query_object/ashmolean_2_query.jpg',1)
print img01.shape
img02 = cv2.imread('C:/Cassandra/python_oxford/database/ashmolean_000000.jpg',1)
cv2.imshow('query', img01)
cv2.imshow('matching image', img02)
cv2.waitKey(0)
cv2.destroyAllWindows()
sift = cv2.SIFT(nfeatures = (img01.shape[0]*img01.shape[1]/238),edgeThreshold=0.01)
kpts01, desc01 = sift.detectAndCompute(img01, None)
sift = cv2.SIFT(nfeatures = (img02.shape[0]*img02.shape[1]/238),edgeThreshold=0.01)
kpts02, desc02 = sift.detectAndCompute(img02, None)
print type(kpts01)
matcher = cv2.BFMatcher(cv2.NORM_L2)
raw_matches = matcher.knnMatch(desc01, trainDescriptors = desc02, k = 2)
nnThreshold = 0.8
good_match = []
for i in range(len(raw_matches)):
    if (raw_matches[i][0].distance / raw_matches[i][1].distance) <= float(pow(nnThreshold,2)):
        good_match.append(raw_matches[i][0])



print 'number of good matches: ', len(good_match)

print good_match[0].distance
minGoodMatch = 10

if len(good_match) >= minGoodMatch:
    # good_match_distance = np.zeros((1,len(good_match)), float)
    # print type(good_match[0].distance)
    # for i in range(len(good_match)):
    #     good_match_distance[0,i] = good_match[i].distance
    #
    # good_match_dis_rank = np.argsort(good_match_distance, axis=1)
    # print good_match_dis_rank
    # print good_match[good_match_dis_rank[0][0]].distance
    # print good_match[good_match_dis_rank[0][-1]].distance
    #
    # chosen_match = []
    # for i in range(10):
    #     chosen_match.append(good_match[good_match_dis_rank[0][i]])

    src_pts = np.reshape(np.float32([ kpts01[m.queryIdx].pt for m in good_match ]),(-1,1,2))
    dst_pts = np.reshape(np.float32([ kpts02[m.trainIdx].pt for m in good_match ]),(-1,1,2))

    print src_pts

    homograph_start = clock()
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print 'M: ', M
    homograph_end = clock()
    print 'Homograph time used: ', homograph_end - homograph_start
    matchesMask = mask.ravel().tolist()

    h,w = img01.shape[0],img01.shape[1]
    pts = np.reshape(np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]),(-1,1,2))
    dst = cv2.perspectiveTransform(pts,M)

    cv2.polylines(img02,[np.int32(dst)],True,(0,0,255),3, 1)
    cv2.imwrite('C:/Cassandra/balliol_1_query_SV.jpg', img02)
    cv2.imshow('query', img01)
    cv2.imshow('img3', img02)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print "Not enough matches are found - %d/%d" % (len(good_match),minGoodMatch)
    matchesMask = None



###########



# good = []
# for m,n in raw_matches:
#     if m.distance < nnThreshold*n.distance:
#         good.append(m)
#
# print len(good)
# if len(good)>100:
#     src_pts = np.float32([ kpts01[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kpts02[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     print type(src_pts)
#     print src_pts.shape
#
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h,w = img01.shape[0],img01.shape[1]
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#
#     # img2 = cv2.polylines(img02,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
# else:
#     print "Not enough matches are found - %d/%d" % (len(good),100)
#     matchesMask = None


# cv2.imshow('img1', img01)
# cv2.imshow('img2', img02)
# cv2.waitKey(0)
# cv2.destroyAllWindows()