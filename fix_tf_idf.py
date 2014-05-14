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
if __name__ == "__main__":
    ## Operation bools:
    read_database_image_dirs = True
    read_desc_csv = True

    ## dirs
    top_dir = 'C:/Cassandra/python_oxford/'
    database_image_dir = top_dir + 'database/'
    database_desc_dir = top_dir + 'database_desc/'
    database_kpts_dir = top_dir + 'database_kpts/'
    database_VW_dir = top_dir + 'database_VW/'
    ## parameters
    cluster_number = 1024
    des_dimension = 128
    result=[]
    result_img_dir =[]
    result_img_kpts = []
    index_file = open(top_dir + 'image_index_python.txt','rb')
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
    # IDF matrix: 1 * 128clusters

    IDF_matrix = np.zeros((1, cluster_number), np.float64)
    VW_showing_up = np.zeros((1, cluster_number), np.int32)

    ##### get the VW_showing_up
    for i in range(len(result_img_dir)):
        the_file = open(database_VW_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_VW.txt','r')
        # jump first two lines
        line = the_file.readline()
        line = the_file.readline()
        # read third line
        line = the_file.readline()
        TF_IDF_tmp = np.zeros((1,cluster_number), np.float64)
        for j in range(cluster_number):
            TF_IDF_tmp[0, j] = np.float64((line.split(','))[j])
        VW_showing_up = VW_showing_up + np.int32(TF_IDF_tmp > 0)

    ################
    for i in range(cluster_number):
        IDF_matrix[0,i] = math.log10(float(len(result_img_dir)) / float(VW_showing_up[0,i]))
    # print IDF_matrix

    print '...starting computing tf-idf matrix...'
    tf_idf_timing_start = clock()


    TF_IDF_out = np.zeros((len(result_img_dir), cluster_number), np.int32)
    for i in range(len(result_img_dir)):
        the_file = open(database_VW_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_VW.txt','r')
        # jump first two lines
        line = the_file.readline()
        line = the_file.readline()
        # read third line
        line = the_file.readline()
        TF_IDF_tmp = np.zeros((1,cluster_number), np.float64)
        for j in range(cluster_number):
            TF_IDF_tmp[0, j] = (0.5 + 0.5 * np.float64((line.split(','))[j]) / np.float64(VW_max_occur[0,i])) * IDF_matrix[0, j]
        # Normalize TF_IDF_tmp
        TF_IDF_inner = math.sqrt(np.dot(TF_IDF_tmp, np.transpose(TF_IDF_tmp)))
        # TF_IDF_tmp = TF_IDF_tmp / TF_IDF_inner
        TF_IDF_tmp /= TF_IDF_inner
        TF_IDF_out[i,:] = TF_IDF_tmp

        the_file.close()
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

    tf_idf_timing_end = clock()
    print '...tf-idf time used: ', int((tf_idf_timing_end - tf_idf_timing_start)/60), ' minutes ', \
        int((tf_idf_timing_end - tf_idf_timing_start)%60), ' seconds...'