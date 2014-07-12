__author__ = 'LIMU_North'

#### 2014/07/08  generate csv file for kmeans in C++


import os
import math
import csv
import gc
import cv2

import numpy as np

from time import gmtime, strftime


## timeit!!!
from time import clock
if __name__ == "__main__":
    ## Operation bools:
    read_database_image_dirs = True
    read_desc_csv = False
    perform_kmeans = True
    bool_generate_VW = True
    # bool_read_VW_txt_for_TF_IDF = False
    ## parameters
    subsampling_num = 800000
    cluster_number = 50000
    des_dimension = 128
    ## dirs
    top_dir = 'C:/Cassandra/python_oxford/'
    database_image_dir = top_dir + 'database/'
    database_desc_dir = top_dir + 'database_desc/'
    database_kpts_dir = top_dir + 'database_kpts/'
    database_VW_dir = top_dir + 'database_VW/'
    try:
        os.stat(database_image_dir)
    except:
        os.mkdir(database_image_dir)
    try:
        os.stat(database_desc_dir)
    except:
        os.mkdir(database_desc_dir)
    try:
        os.stat(database_kpts_dir)
    except:
        os.mkdir(database_kpts_dir)
    try:
        os.stat(database_VW_dir)
    except:
        os.mkdir(database_VW_dir)

    result=[]
    result_img_dir =[]
    result_img_kpts = []
    #################
    start=clock()
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
    ## 14/04/28 change the way of making des_mat
    ## gc.collect()??
    gc.collect()
    print 'total_des_count ', total_des_count
    des_mat = np.zeros((total_des_count, des_dimension), np.float32)
    des_count_present = 0

    ## read descriptors from csv files.
    if read_desc_csv:
        time_read_csv_start = clock()
        for i in range(len(result_img_dir)):
            print '...reading des csv of image number ', i+1 , '...'
            img_des_tmp = database_desc_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_des.csv'
            the_file = open(img_des_tmp,'rb')
            # des_mat_tmp = np.zeros((result_img_kpts[i], des_dimension), np.int32)
            reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)
            # reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)

            for row in reader:
                des_mat[des_count_present,:] = np.array(map(np.float32, row))
                des_count_present += 1
            the_file.close()
        print des_mat.shape
        time_read_csv_end = clock()
        print 'read csv time used: ', int((time_read_csv_end - time_read_csv_start)/60), ' minutes ', \
            int((time_read_csv_end - time_read_csv_start)%60), ' seconds...'
        print
        print '...Saving des_mat into npy...'
        start_npy = clock()
        np.save(top_dir + 'des_mat.npy',des_mat)
        end_npy = clock()
        print '...Saving npy time used: ', end_npy - start_npy, ' seconds...'
    else:
        load_des_mat_start = clock()
        des_mat = np.load(top_dir + 'des_mat.npy')
        load_des_mat_end = clock()
        print '...des_mat reloaded...time used: ', load_des_mat_end - load_des_mat_start, ' seconds...'
        print '...des_mat shape: ', des_mat.shape[0], ' ', des_mat.shape[1], ' ...'
        print

    ## here we do a sub-sampling

    randn_percentage = float(subsampling_num) / des_mat.shape[0]
    myrandn = np.random.uniform(low=0.0, high=1.0, size=des_mat.shape[0])
    mysubset = tuple(np.where(myrandn<= randn_percentage)[0])

    ## our des_mat subset
    des_mat_subset = des_mat[mysubset, :]
    ## delete des_mat
    des_mat = None


    print 'starting output...'
    output_csv_file = open(top_dir + 'output_csv_file.csv', 'w')
    for i in range(des_mat_subset.shape[0]):
        for j in range(des_mat_subset.shape[1]):
            output_csv_file.write(str(des_mat_subset[i,j]))
            if j < (des_mat_subset.shape[1]-1):
                output_csv_file.write(',')
        output_csv_file.write('\n')
    output_csv_file.close()
    print 'output finished...'

