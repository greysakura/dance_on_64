__author__ = 'LIMU_North'

import cv2
import numpy as np
import sys

img_break = cv2.imread('C:/Cassandra/oxbuild_images/all_souls_000002.jpg',1)
sift = cv2.SIFT(edgeThreshold=0.01)
# sift = cv2.SIFT(contrastThreshold=0.01)
kpts, desc = sift.detectAndCompute(img_break, None)
cv2.drawKeypoints(img_break,kpts,img_break,(255,0,0),1)
print len(kpts)


cv2.imshow('break', img_break)
cv2.waitKey(0)
cv2.destroyAllWindows()