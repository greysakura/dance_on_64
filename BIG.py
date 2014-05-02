__author__ = 'LIMU_North'

import cv2
import sys
import os
import math
import csv
import numpy as np
import gc
## timeit!!!
from time import clock
start=clock()

top_dir = 'C:/Cassandra/here/'
cluster_number = 8192
# Using SIFT here
#
des_dimension = 128
result=[]
result_img_dir =[]
result_img_kpts = []
index_dir = top_dir + 'image_index_python.txt'
index_file = open(index_dir,'rb')
# not used here, but will be used.
# line = index_file.readline()
total_des_count = 0
for line in index_file:
    result_img_dir.append((line.split(','))[0])
    result_img_kpts.append(int(float(line.split(',')[1][:-2])))
    total_des_count += int(float(line.split(',')[1][:-2]))
print result_img_dir
print result_img_kpts
print 'total_des_count: ', total_des_count
index_file.close()
image_count = len(result_img_dir)
## 14/04/28 change the way of making des_mat

## time it
read_csv_start = clock()

des_mat = np.zeros((total_des_count, des_dimension), np.int32)
des_count_present = 0
for i in range(len(result_img_dir)):
    img_des_tmp = (result_img_dir[i].split('.'))[0] + '_des.csv'
    print img_des_tmp
    the_file = open(img_des_tmp,'rb')
    # des_mat_tmp = np.zeros((result_img_kpts[i], des_dimension), np.int32)
    reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)
    # reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)

    for row in reader:
        for j in range(len(row)):
            des_mat[des_count_present, j] = int(float(row[j]))
            # except:
            #     print 'y: ', row_count, ' x: ', i
        des_count_present += 1


    the_file.close()
print des_mat.shape

read_csv_end = clock()
csv_time_used = read_csv_end - read_csv_start

print 'Reading csv, time used: ', csv_time_used, ' seconds.'
##
img1 = cv2.imread('C:/Cassandra/all_souls_000091.jpg')
cv2.imshow('ready for Kmean?', img1)
print 'ARE YOU READY!!!'
cv2.waitKey(0)
cv2.destroyAllWindows()


# set as float32
des_mat = np.float32(des_mat)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
# We use 128 clusters for our K-means clustering.
# Define criteria_kmeans = ( type, max_iter = 10 , epsilon = 1.0 )
criteria_kmeans = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 1)

start_kmeans = clock()

# Apply KMeans  cv2.kmeans(data, K, criteria, attempts, flags[, bestLabels[, centers]])
compactness,labels,centers = cv2.kmeans(data= des_mat, K = cluster_number, bestLabels=None,
                                        criteria= criteria_kmeans, attempts=5,flags=cv2.KMEANS_RANDOM_CENTERS)
finish_kmeans = clock()

print compactness

print 'centers: ', centers

print type(centers[0,0])
print centers.shape

kmeans_result_append = '/kmeans_result.txt'
kmeans_result_dir = top_dir + kmeans_result_append
kmeans_result_file = open(kmeans_result_dir, 'w')

kmeans_result_file.write(str(cluster_number))
kmeans_result_file.write(',')
kmeans_result_file.write(str(des_dimension))
kmeans_result_file.write(',')
kmeans_result_file.write(str(compactness))
kmeans_result_file.write('\n')

for i in range(centers.shape[0]):
    for j in range(centers.shape[1]):
        kmeans_result_file.write(str(centers[i,j]))
        if j < (centers.shape[1]-1):
            kmeans_result_file.write(',')
    kmeans_result_file.write('\n')

kmeans_result_file.close()

# print labels
print 'label numbers: ', len(labels)
# print type(labels[0][0])

label_size = np.zeros((1, cluster_number), np.int32)
for i in range(len(labels)):
    label_size[0, labels[i][0]] +=1

print label_size
print 'compactness: ', compactness

# Generate Visual Word

des_count_for_VW = 0
VW_max_occur = np.zeros((1,len(result_img_dir)),np.int32)
VW_showing_up = np.zeros((1, cluster_number), np.int32)

## prepare an empty inverted_file_matrix. But the size is not (0,0)
# inverted_file_matrix = np.zeros((cluster_number,0), np.int32)
## change 04/30 use a big one.
inverted_file_matrix = np.zeros((cluster_number, len(result_img_dir)), np.int32)
inverted_file_matrix[:,1] = np.ones((cluster_number,1),np.int32)

##

for i in range(len(result_img_dir)):

    img_des_tmp = (result_img_dir[i].split('.'))[0] + '_VW.txt'
    the_file = open(img_des_tmp,'w')
    the_file.write(str(result_img_kpts[i]))
    the_file.write(',')
    the_file.write(str(cluster_number))
    the_file.write('\n')

    # VW of present image.
    VW_tmp = np.zeros((1,cluster_number),np.int32)
    for j in range(result_img_kpts[i]):
        VW_tmp[0,labels[j + des_count_for_VW][0]] += 1
        the_file.write(str(labels[j + des_count_for_VW][0]))
        if (result_img_kpts>=1) & (j < (result_img_kpts[i] - 1)):
            the_file.write(',')
    the_file.write('\n')
    ## Extra: for inverted file
    inverted_file_matrix[:,i] = (np.int32(VW_tmp.transpose() > 0))

    # inverted_file_matrix = np.concatenate((inverted_file_matrix, np.int32(VW_tmp.transpose() > 0)), axis = 1)

    for j in range(cluster_number):
        the_file.write(str(VW_tmp[0,j]))
        if j < cluster_number - 1:
            the_file.write(',')

        if VW_tmp[0,j] != 0:
            VW_showing_up[0,j] += 1
    the_file.write('\n')
    # print VW_tmp.sum(dtype=np.int32)
    des_count_for_VW += result_img_kpts[i]
    # position where we can find the max
    # print VW_tmp.argmax(axis = 1)[0]
    # the max value
    # print VW_tmp[0,VW_tmp.argmax(axis = 1)][0]
    # the max value, another version
    # print np.amax(VW_tmp, axis=1)[0]
    VW_max_occur[0,i] = np.amax(VW_tmp, axis=1)[0]
    the_file.close()

# IDF matrix: 1 * 128clusters

IDF_matrix = np.zeros((1, cluster_number), np.float64)

for i in range(cluster_number):
    IDF_matrix[0,i] = math.log10(float(len(result_img_dir)) / float(VW_showing_up[0,i]))
# print IDF_matrix


TF_IDF_matrix = np.zeros((1,len(labels)), np.float64)

for i in range(len(result_img_dir)):
    img_des_tmp = (result_img_dir[i].split('.'))[0] + '_VW.txt'
    the_file = open(img_des_tmp,'r')
    # jump first two lines
    line = the_file.readline()
    line = the_file.readline()
    # read third line
    line = the_file.readline()
    TF_IDF_tmp = np.zeros((1,cluster_number), np.float64)
    for j in range(cluster_number):
        TF_IDF_tmp[0, j] = (0.5 + 0.5 * float((line.split(','))[j]) / float(VW_max_occur[0,i])) * IDF_matrix[0, j]
    # Normalize TF_IDF_tmp
    TF_IDF_inner = math.sqrt(np.dot(TF_IDF_tmp, np.transpose(TF_IDF_tmp)))
    TF_IDF_tmp = TF_IDF_tmp / TF_IDF_inner
    if i == 0:
        TF_IDF_out = TF_IDF_tmp.copy()
    else:
        TF_IDF_out = np.concatenate((TF_IDF_out, TF_IDF_tmp), axis = 0)

TF_IDF_append = '/TF_IDF_matrix.txt'
TF_IDF_dir = top_dir + TF_IDF_append
TF_IDF_file = open(TF_IDF_dir, 'w')
TF_IDF_file.write(str(TF_IDF_out.shape[0]))
TF_IDF_file.write(',')
TF_IDF_file.write(str(TF_IDF_out.shape[1]))
TF_IDF_file.write('\n')

for i in range(TF_IDF_out.shape[0]):
    for j in range(TF_IDF_out.shape[1]):
        TF_IDF_file.write(str(TF_IDF_out[i,j]))
        if j < (TF_IDF_out.shape[1] - 1):
            TF_IDF_file.write(',')
    TF_IDF_file.write('\n')
TF_IDF_file.close()

print 'inverted file matrix: '
print inverted_file_matrix
print inverted_file_matrix.shape
## write inverted file storage file
inverted_file_append = '/inverted_file_matrix_python.txt'
inverted_file_dir = top_dir + inverted_file_append
inverted_file_file = open(inverted_file_dir, 'w')
## let's write it.
inverted_file_file.write(str(cluster_number))
inverted_file_file.write(',')
inverted_file_file.write(str(image_count))
inverted_file_file.write('\n')
for i in range(inverted_file_matrix.shape[0]):
    for j in range(inverted_file_matrix.shape[1]):
        inverted_file_file.write(str(inverted_file_matrix[i,j]))
        if j < (inverted_file_matrix.shape[1] - 1):
            inverted_file_file.write(',')
    inverted_file_file.write('\n')
inverted_file_file.close()


print 'number of images: '
print image_count

print centers.shape

finish=clock()

print 'total time used: ', (finish-start)
print 'kmean time used: ', finish_kmeans - start_kmeans