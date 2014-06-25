__author__ = 'LIMU_North'

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
    perform_kmeans = False
    bool_generate_VW = True
    # bool_read_VW_txt_for_TF_IDF = False
    ## parameters
    cluster_number = 10000
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

    randn_percentage = 800000.0 / des_mat.shape[0]
    myrandn = np.random.uniform(low=0.0, high=1.0, size=des_mat.shape[0])
    mysubset = tuple(np.where(myrandn<= randn_percentage)[0])

    ## our des_mat subset
    des_mat_subset = des_mat[mysubset, :]
    ## delete des_mat
    des_mat = None
    # raw_input('stop here.')
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # We use 128 clusters for our K-means clustering.
    # Define criteria_kmeans = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria_kmeans = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 3)
    print '...starting kmeans...'

    compactness = 0
    if perform_kmeans:
        # Apply KMeans  cv2.kmeans(data, K, criteria, attempts, flags[, bestLabels[, centers]])
        start_kmeans = clock()
        compactness,labels,centers = cv2.kmeans(data= des_mat_subset, K = cluster_number, bestLabels=None,
                                                criteria= criteria_kmeans, attempts=5,flags=cv2.KMEANS_RANDOM_CENTERS)
        finish_kmeans = clock()

        print 'compactness', compactness
        # print 'centers: ', centers

        print 'kmean time used: ', int((finish_kmeans - start_kmeans)/60), 'minutes ', int((finish_kmeans - start_kmeans)%60), ' seconds.'
        print
        print '...Saving labels and centers into npy...'
        start_npy = clock()
        np.save(top_dir + 'kmeans_compactness.npy', compactness)
        np.save(top_dir + 'kmeans_labels.npy',labels)
        np.save(top_dir + 'kmeans_centers.npy',centers)
        end_npy = clock()

        print '...Saving labels and centers npy time used: ', end_npy - start_npy, ' seconds...'
    else:
        labels = np.load(top_dir + 'kmeans_labels.npy')
        centers = np.load(top_dir + 'kmeans_centers.npy')
        compactness = np.load(top_dir + 'kmeans_compactness.npy')
        print '...Loading kmeans result finished...'

    print type(des_mat_subset[0,0])
    timenow = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print timenow
    # raw_input('stop here.')
    print
    print 'Writing kmeans result.'

    write_kmeans_result_start = clock()
    kmeans_result_append = '/kmeans_result.txt'
    kmeans_result_dir = top_dir + kmeans_result_append
    kmeans_result_file = open(top_dir + '/kmeans_result.txt', 'w')

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

    write_kmeans_result_end = clock()
    print 'write kmeans time used: ', int((write_kmeans_result_end - write_kmeans_result_start)/60), ' minutes ' , \
        int((write_kmeans_result_end - write_kmeans_result_start)%60), ' seconds...'

    # print labels
    print 'label numbers: ', len(labels)
    # print type(labels[0][0])

    label_size = np.zeros((1, cluster_number), np.int32)

    # Generate Visual Word
    if bool_generate_VW:
        des_count_for_VW = 0
        ## max frequncy: VW_max_occur
        max_freq = np.zeros((1,cluster_number),np.float64)
        ## document frequency
        document_freq = np.zeros((1, cluster_number), np.int32)

        ## prepare an empty inverted_file_matrix. But the size is not (0,0)

        inverted_file_matrix = np.zeros((cluster_number,len(result_img_dir)), np.int32)
        ##
        ## WRITE ALL VW INTO A SINGLE TXT
        big_VW_file = open(top_dir + 'total_VW.txt', 'w')
        big_VW_norm_file = open(top_dir + 'total_VW_norm.txt', 'w')

        ## create a total VW mat
        total_VW = np.zeros((image_count, cluster_number), np.int32)

        for i in range(len(result_img_dir)):
            ## added for sub-sampling
            des_mat_tmp = np.zeros((result_img_kpts[i], des_dimension), np.float32)

            print '...open des csv of image number ', i , '...'
            img_des_tmp = database_desc_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_des.csv'
            the_file = open(img_des_tmp,'rb')
            # des_mat_tmp = np.zeros((result_img_kpts[i], des_dimension), np.int32)
            reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)
            # reader = csv.reader(the_file, delimiter=',', quoting = csv.QUOTE_NONE)


            des_count_tmp = 0
            for row in reader:
                des_mat_tmp[des_count_tmp,:] = np.array(map(np.float32, row))
                des_count_tmp += 1
            the_file.close()

            # raw_input('stop')
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = matcher.knnMatch(des_mat_tmp, trainDescriptors = centers, k = 1)
            labels_tmp = np.array([m[0].trainIdx for m in raw_matches], np.int32)





            print '...creating VW for image number: ', i, ' ...'
            img_des_tmp = database_VW_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_VW.txt'
            the_file = open(img_des_tmp,'w')
            the_file.write(str(result_img_kpts[i]))
            the_file.write(',')
            the_file.write(str(cluster_number))
            the_file.write('\n')

            # VW of present image.

            VW_tmp = np.zeros((1,cluster_number),np.int32)
            for label_i in range(result_img_kpts[i]):
                VW_tmp[0,labels_tmp[label_i]] += 1
                the_file.write(str(labels_tmp[label_i]))
                if (result_img_kpts>=1) & (label_i < (result_img_kpts[i] - 1)):
                    the_file.write(',')
            the_file.write('\n')
            print 'VW_tmp: ', VW_tmp
            print VW_tmp.sum()
            total_VW[i,:] = VW_tmp
            label_size = label_size + VW_tmp


            ## Extra: for inverted file
            # inverted_file_matrix = np.concatenate((inverted_file_matrix, np.int32(VW_tmp.transpose() > 0)), axis = 1)
            inverted_file_matrix[:,i] = (np.int32(VW_tmp.transpose() > 0)).flatten()


            ## document_freq: used in tf-idf
            document_freq = document_freq + np.int32(VW_tmp > 0)
            des_count_for_VW += result_img_kpts[i]

            ## normalize the VW:
            ## change 14/05/13
            VW_tmp_norm = np.float64(VW_tmp)/(VW_tmp.sum(axis=1)[0])

            max_freq[0,i] = np.amax(VW_tmp, axis=1)[0]
            ## line 3, write the normed VW
            for j in range(cluster_number):
                the_file.write(str(VW_tmp_norm[0,j]))
                big_VW_norm_file.write(str(VW_tmp_norm[0,j]))
                if j < cluster_number - 1:
                    the_file.write(',')
                    big_VW_norm_file.write(',')
            the_file.write('\n')
            big_VW_norm_file.write('\n')

            ## line 4, write the original VW
            for j in range(cluster_number):
                the_file.write(str(VW_tmp[0,j]))
                big_VW_file.write(str(VW_tmp[0,j]))
                if j < cluster_number - 1:
                    the_file.write(',')
                    big_VW_file.write(',')

                # if VW_tmp[0,j] != 0:
                #     document_freq[0,j] += 1
            the_file.write('\n')
            big_VW_file.write('\n')
            the_file.close()
        big_VW_file.close()
        big_VW_norm_file.close()


        ## Save the max occur
        max_freq_file = open(top_dir + 'max_freq.txt', 'w')
        for write_i in range(max_freq.shape[0]):
            for write_j in range(max_freq.shape[1]):
                max_freq_file.write(str(max_freq[write_i,write_j]))
                if write_j < (max_freq.shape[1]-1):
                    max_freq_file.write(',')
            max_freq_file.write('\n')
        max_freq_file.close()

        document_freq_file = open(top_dir + 'document_freq.txt', 'w')
        for write_i in range(document_freq.shape[0]):
            for write_j in range(document_freq.shape[1]):
                document_freq_file.write(str(document_freq[write_i,write_j]))
                if write_j < (document_freq.shape[1]-1):
                    document_freq_file.write(',')
            document_freq_file.write('\n')
        document_freq_file.close()

        ## write inverted file storage file
        inverted_file_file = open(top_dir + 'inverted_file_matrix.txt', 'w')
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

        np.save(top_dir + 'total_VW.npy', total_VW)
        np.save(top_dir + 'max_freq.npy',max_freq)
        np.save(top_dir + 'document_freq.npy', document_freq)
        np.save(top_dir + 'inverted_file_matrix.npy', inverted_file_matrix)
    else:
        total_VW = np.load(top_dir + 'total_VW.npy')
        max_freq = np.load(top_dir + 'max_freq.npy')
        document_freq = np.load(top_dir + 'document_freq.npy')
        inverted_file_matrix = np.load(top_dir + 'inverted_file_matrix.npy')

        print 'total_VW.shape: ', total_VW.shape
        print 'max_freq.shape: ', max_freq.shape
        print 'document_freq.shape: ', document_freq.shape
        print 'inverted_file_matrix.shape: ', inverted_file_matrix.shape


    raw_input('wait here.')
    IDF_matrix = np.zeros((1, cluster_number), np.float64)
    ## change 06/20
    for i in range(cluster_number):
        IDF_matrix[0,i] = math.log10(float(len(result_img_dir)) / float(document_freq[0,i]))
        # IDF_matrix[0,i] = math.log10(float(len(result_img_dir)) / float(1 + document_freq[0,i]))
    print IDF_matrix
    print '...starting computing tf-idf matrix...'
    tf_idf_timing_start = clock()
    TF_IDF_matrix = np.zeros((1,len(labels)), np.float64)
    IDF_matrix_tmp = IDF_matrix.flatten()
    TF_IDF_out = np.zeros((len(result_img_dir), cluster_number), np.float64)
    for i in range(len(result_img_dir)):
        print 'TF_IDF for number: ', i+1
        # img_des_tmp = database_VW_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_VW.txt'
        the_file = open(database_VW_dir + ((result_img_dir[i].split('.'))[0]).split('/')[-1] + '_VW.txt','r')
        # jump first two lines
        line = the_file.readline()
        line = the_file.readline()
        line = the_file.readline()
        # read fourth line
        line = the_file.readline()
        the_file.close()
        VW_read = np.float64(line.split(','))
        TF_IDF_tmp = np.zeros((1,cluster_number), np.float64)
        # print type(line.split(','))
        # print np.max(VW_read)


        max_freq_tmp = np.ones((cluster_number,), np.float64) * max_freq[0,i]
        TF_IDF_tmp[0,:] = VW_read * IDF_matrix_tmp / max_freq_tmp
        # print np.max(TF_IDF_tmp)
        # raw_input('stop')
        # TF_IDF_tmp[0,:] =

        ## here we generate the tf-idf
        # for j in range(cluster_number):
        #     TF_IDF_tmp[0, j] = (np.float64((line.split(','))[j]) / np.float64(max_freq[0,i])) * IDF_matrix[0, j]
            # TF_IDF_tmp[0, j] = (0.5 + 0.5 * np.float64((line.split(','))[j]) / np.float64(max_freq[0,i])) * IDF_matrix[0, j]
        # Normalize TF_IDF_tmp
        # TF_IDF_inner = math.sqrt(np.dot(TF_IDF_tmp, np.transpose(TF_IDF_tmp)))
        # TF_IDF_tmp = TF_IDF_tmp / TF_IDF_inner
        # TF_IDF_tmp /= TF_IDF_inner
        TF_IDF_out[i,:] = np.copy(TF_IDF_tmp)

    print 'here!'
    ## write TF_IDF record matrix txt.
    TF_IDF_append = 'TF_IDF_matrix.txt'
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
    np.save(top_dir + 'TF_IDF_matrix.npy', TF_IDF_out)
    ## write IDF record matrix txt for later use.
    IDF_file = open(top_dir + 'IDF_matrix.txt', 'w')
    IDF_file.write(str(IDF_matrix.shape[0]))
    IDF_file.write(',')
    IDF_file.write(str(IDF_matrix.shape[1]))
    IDF_file.write('\n')

    for IDF_i in range(IDF_matrix.shape[0]):
        for IDF_j in range(IDF_matrix.shape[1]):
            IDF_file.write(str(IDF_matrix[IDF_i, IDF_j]))
            if IDF_j < (IDF_matrix.shape[1] - 1):
                IDF_file.write(',')
        IDF_file.write('\n')
    IDF_file.close()
    np.save(top_dir + 'IDF_matrix.npy', IDF_matrix)
    #####
    tf_idf_timing_end = clock()
    print '...tf-idf time used: ', int((tf_idf_timing_end - tf_idf_timing_start)/60), ' minutes ', \
        int((tf_idf_timing_end - tf_idf_timing_start)%60), ' seconds...'



    print
    print '...total number of images: ', image_count, ' ...'
    finish=clock()
    print '...total time used: ', int((finish-start)/60), ' minutes ', int((finish - start)%60), ' seconds...'
