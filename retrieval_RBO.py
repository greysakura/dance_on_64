__author__ = 'LIMU_North'

import cv2
import numpy as np
import os
import math
from inversion_test import easy_merge_sort
from sklearn import svm, linear_model
from time import clock

def intersection_check(list01, list02, d):
    if (type(list01) is list) and (type(list02) is list):
    #### d is the depth of checking
        list01_set = [(list01[i]) for i in range(d)]
        list02_set = [(list02[i]) for i in range(d)]
        intersection = set(list01_set) & set(list02_set)
        return len(intersection)
    return 0

def rank_biased_overlap(list01,list02, p, h):
    if (type(list01) is list) and (type(list02) is list):
        RBO = 0.0
        for d in range(1, h+1):
            RBO += (1-p) * math.pow(p,d-1) * intersection_check(list01,list02,d) / d
        return RBO
    return 0


if __name__ == "__main__":
    top_dir = 'C:/Cassandra/python_oxford/'
    database_image_dir = top_dir + 'database/'
    database_desc_dir = top_dir + 'database_desc/'
    database_kpts_dir = top_dir + 'database_kpts/'
    database_VW_dir = top_dir + 'database_VW/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'
    TF_IDF_ranking_dir = top_dir + 'TF_IDF_ranking/'
    SV_result_dir = top_dir + 'SV_verified/'
    SV_reranking_dir = top_dir + 'SV_reranking/'
    RBO_reranking_dir = top_dir + 'RBO_reranking/'

    DQE_reranking_dir = top_dir + 'DQE_reranking/'

    try:
        os.stat(TF_IDF_ranking_dir)
    except:
        os.mkdir(TF_IDF_ranking_dir)
    try:
        os.stat(SV_result_dir)
    except:
        os.mkdir(SV_result_dir)
    try:
        os.stat(SV_reranking_dir)
    except:
        os.mkdir(SV_reranking_dir)
    try:
        os.stat(DQE_reranking_dir)
    except:
        os.mkdir(DQE_reranking_dir)
    try:
        os.stat(RBO_reranking_dir)
    except:
        os.mkdir(RBO_reranking_dir)


    ## operation bools
    bool_using_tf_idf = True
    bool_read_tf_idf_from_txt = False
    bool_read_idf_from_txt = False
    bool_read_database_VW_from_txt = False

    # Number of clusters: 128 at present
    cluster_number = 50000
    first_retrieval_num = 5062
    # Using SIFT here
    des_dimension = 128
    kpts_density = 238
    # For SV
    num_for_SV = 200
    nnThreshold = 0.8
    minGoodMatch = 10
    ## store images' dirs
    query_img_dir_list = []
    ## list inside a list
    query_img_matching_img_list = []
    # target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
    query_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
    for line in query_img_list:
        query_img_dir_list.append(line[:-1])
        # print line[:-1]
    query_img_list.close()
    print 'len(query_img_dir_list): ', len(query_img_dir_list)

    ## read kmeans centers
    kmeans_result_file = open(top_dir + 'kmeans_result.txt', 'r')

    line = kmeans_result_file.readline()
    centers = np.zeros((cluster_number, des_dimension), np.float32)
    read_kmeans_time_start = clock()
    for i in range(cluster_number):
        line = kmeans_result_file.readline()
        centers[i,:] = np.float32(line.split(','))
    kmeans_result_file.close()
    read_kmeans_time_end = clock()
    print 'read kmeans time used: ', read_kmeans_time_end - read_kmeans_time_start
    print
    # read the TF_IDF_matrix from file.
    # prepare for further VW matching under tf-idf structure

    print '...reading TF-IDF matrix...'
    read_tf_idf_start = clock()
    if bool_read_tf_idf_from_txt:
        TF_IDF_append = 'TF_IDF_matrix.txt'
        TF_IDF_dir = top_dir + TF_IDF_append
        TF_IDF_file = open(TF_IDF_dir, 'r')
        line = TF_IDF_file.readline()
        # TF_IDF_matrix
        TF_IDF_matrix = np.zeros((int(line.split(',')[0]), int(line.split(',')[1])), np.float64)
        for i in range(TF_IDF_matrix.shape[0]):
            line = TF_IDF_file.readline()
            TF_IDF_matrix[i,:] = np.float64(line.split(','))
        ## print TF_IDF_matrix
        TF_IDF_file.close()
    else:
        TF_IDF_matrix = np.load(top_dir + 'TF_IDF_matrix.npy')
    print 'TF_IDF_matrix: ', TF_IDF_matrix.shape

    # # ## urgent task
    # TF_IDF_norm_matrix = np.zeros(TF_IDF_matrix.shape, np.float64)
    # for i in range(TF_IDF_matrix.shape[0]):
    #     tmp_tf_idf = TF_IDF_matrix[i,:]
    #     tmp_tf_idf /= math.sqrt(np.dot(tmp_tf_idf.flatten(), tmp_tf_idf.flatten()))
    #     TF_IDF_norm_matrix[i,:] = tmp_tf_idf
    # np.save(top_dir + 'TF_IDF_norm_matrix.npy', TF_IDF_norm_matrix)
    # # raw_input('stop!!!!!!!!!!!')

    TF_IDF_norm_matrix = np.load(top_dir + 'TF_IDF_norm_matrix.npy')


    read_tf_idf_end = clock()

    print '...TF-IDF reading time: ', read_tf_idf_end - read_tf_idf_start, 'seconds...'
    print

    ## read IDF_matrix
    print '...reading IDF matrix...'
    if bool_read_idf_from_txt:
        IDF_file = open(top_dir + 'IDF_matrix.txt','r')
        line = IDF_file.readline()
        IDF_matrix = np.zeros((int(line.split(',')[0]), int(line.split(',')[1])), np.float64)
        for IDF_i in range(IDF_matrix.shape[0]):
            line = IDF_file.readline()
            IDF_matrix[IDF_i,:] = np.float64(line.split(','))
        IDF_file.close()
    else:
        IDF_matrix = np.load(top_dir + 'IDF_matrix.npy')

    print '...IDF matrix load finished...'
    print

    result_img_dir =[]
    result_img_kpts = []
    print '...reading database image dirs...'
    index_file = open(top_dir + 'image_index_python.txt','rb')
    image_count = 0
    for line in index_file:
        result_img_dir.append((line.split(','))[0])
        result_img_kpts.append(int(float(line.split(',')[1][:-2])))
    # print result_img_dir
    # print result_img_kpts
    index_file.close()
    image_count = len(result_img_dir)
    print '...database image dirs loaded...'
    print

    ## pre-load all database VWs into memory
    print '...reading all database image VWs into memory...'
    database_VW_matrix = np.zeros((len(result_img_dir),cluster_number), np.float64)
    if bool_read_database_VW_from_txt:
        for i in range(len(result_img_dir)):
            the_file = open(database_VW_dir + ((result_img_dir[i].split('/'))[-1]).split('.')[0] + '_VW.txt','r')
            line = the_file.readline()
            line = the_file.readline()
            # read the third line
            line = the_file.readline()
            # get the VW of database image
            database_VW_matrix[i,:] = np.array(map(np.float64,line.split(',')))
            # print type(VW_tmp)
            the_file.close()
        np.save(top_dir + 'database_VW_matrix.npy', database_VW_matrix)
    else:
        database_VW_matrix = np.load(top_dir + 'total_VW.npy')


    print '...database image VWs loaded...'
    print

    ## load IDF_matrix
    IDF_matrix = np.load(top_dir + 'IDF_matrix.npy')

    retrieval_time_used_start = clock()

    ## 07/15 record on how many support vector used.
    num_support_vector_list = []


    ## 14/06/14
    SV_time_list = []
    SV_got_list = []

    DQE_predict_time_list = []

    DQE_train_list = []
    ##change 14/05/01
    for query_i in range(len(query_img_dir_list)):
        # tmp_img_matching_list_file = open(query_img_dir_list[query_i][:-3] + 'txt', 'w')
        tmp_img_matching_list_file = open(TF_IDF_ranking_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_TF_IDF_ranking.txt','w')
        tmp_img_matching_img_list = []
        # print query_img_dir_list[query_i]
        ## import target image
        img = cv2.imread(query_img_dir_list[query_i])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        num_feature = int(img_gray.shape[0] * img_gray.shape[1] / kpts_density)
        sift = cv2.SIFT(nfeatures=num_feature, edgeThreshold=0.01)
        kpts_query, desc_query = sift.detectAndCompute(img_gray, None)
        print 'kpts numbers of ', query_img_dir_list[query_i], ' : ', len(kpts_query)

        ## 06/20 new: use BFmatcher
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(desc_query, trainDescriptors = centers, k = 1)
        query_image_keypoint_labels = np.array([m[0].trainIdx for m in raw_matches], np.int32)

        ## generate VW for target image.
        query_image_VW = np.zeros((1, cluster_number), np.float64)
        for i in range(query_image_keypoint_labels.shape[0]):
            query_image_VW[0, query_image_keypoint_labels[i]] += 1.0
        print 'query_image_VW: ', query_image_VW[0,:]

        # ## new image's descriptor file output
        that_file = open(query_img_dir_list[query_i][:-4] + '_des.csv', 'w')
        for i in range(desc_query.shape[0]):
            for j in range(desc_query.shape[1]):
                that_file.write(str(desc_query[i, j]))
                if j < (desc_query.shape[1]-1):
                    that_file.write(',')
            that_file.write('\n')
        that_file.close()

        # new image's kpts file output

        that_file = open(query_img_dir_list[query_i][:-4] + '_kpts.csv', 'w')
        for i in range(len(kpts_query)):
            that_file.write(str(kpts_query[i].pt[0]))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].pt[1]))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].size))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].angle))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].response))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].octave))
            that_file.write(str(','))
            that_file.write(str(kpts_query[i].class_id))
            that_file.write('\n')
        that_file.close()
        #
        ## new image's VW file output
        query_image_VW_file = open(query_img_dir_list[query_i][:-4] + '_VW.txt', 'w')
        query_image_VW_file.write(str(len(kpts_query)))
        query_image_VW_file.write(',')
        query_image_VW_file.write(str(cluster_number))
        query_image_VW_file.write('\n')
        for i in range(query_image_keypoint_labels.shape[0]):
            query_image_VW_file.write(str(query_image_keypoint_labels[i]))
            if i < (query_image_keypoint_labels.shape[0] - 1):
                query_image_VW_file.write(',')
        query_image_VW_file.write('\n')
        for i in range(query_image_VW.shape[1]):
            query_image_VW_file.write(str(query_image_VW[0,i]))
            if i < (query_image_VW.shape[1] - 1):
                query_image_VW_file.write(',')
        query_image_VW_file.write('\n')
        query_image_VW_file.close()
        ## 06/20 new: tf-idf for query image
        # query_TF_IDF = np.float64((0.5 + 0.5 * np.float64(query_image_VW)/(np.float64(query_image_VW.max(axis = 1)[0]))) * IDF_matrix)
        query_TF_IDF = np.float64(np.float64(query_image_VW)/(np.float64(query_image_VW.max(axis = 1)[0])) * IDF_matrix)
        query_TF_IDF_norm = query_TF_IDF / math.sqrt(np.dot(query_TF_IDF.flatten(), query_TF_IDF.flatten()))

        print 'query tf-idf: ', query_TF_IDF.shape
        # raw_input('stop')


        # ##########
        #
        # # now we calculate the "distance" between each database image and target image
        ## 2014/06/25  something to do the inverted file part.
        L2_distance_between_image = np.zeros((1,image_count), np.float64)
        # query_image_VW_norm = query_image_VW / query_image_VW.sum()
        ## Use the right tf-idf Matrix!!!!!!!!  14/04/28
        for i in range(len(result_img_dir)):
            TF_IDF_eye = np.reshape(TF_IDF_matrix[i,:],(1,-1))
            TF_IDF_eye = TF_IDF_eye / math.sqrt(np.dot(TF_IDF_eye.flatten(), TF_IDF_eye.flatten()))
            ##14/05/05  normalize the VW, and then calculate distance.

            # VW_tmp_norm = VW_tmp / VW_tmp.sum()
            if bool_using_tf_idf:
                L2_distance_between_image[0, i] = np.dot((TF_IDF_eye.flatten() - query_TF_IDF_norm.flatten()),(TF_IDF_eye.flatten() - query_TF_IDF_norm.flatten()))
                # L2_distance_between_image[0, i] = np.dot(TF_IDF_eye.flatten(),query_TF_IDF.flatten()) / (math.sqrt(np.dot(TF_IDF_eye.flatten(), TF_IDF_eye.flatten())) * math.sqrt(np.dot(query_TF_IDF.flatten(), query_TF_IDF.flatten())))
            else:
                L2_distance_between_image[0, i] = np.dot((TF_IDF_eye.flatten() - query_TF_IDF.flatten()),(TF_IDF_eye.flatten() - query_TF_IDF.flatten()))
                # L2_distance_between_image[0, i] = np.dot((np.float64(query_image_VW - database_VW_matrix[i,:])), np.transpose(np.float64(query_image_VW - database_VW_matrix[i,:])))

            # L2_distance_between_image[0, i] = np.dot((np.multiply(np.float64(query_image_VW - VW_tmp), TF_IDF_eye)),
            #                                       np.transpose(np.float64(query_image_VW - VW_tmp)))
        print len(np.where(L2_distance_between_image == 0)[1])
        distance_ranking = np.argsort(L2_distance_between_image, axis=1)
        ## this ranking need to be inverse first
        print distance_ranking[0]
        first_ranking = list(distance_ranking[0])
        # raw_input('stop')
        ranked_result_name_dir = []
        # raw_input('stop here')
        for i in range(first_retrieval_num):

            ranked_result_name_dir.append((result_img_dir[distance_ranking[0][i]].split('.')[0]).split('/')[-1])
            tmp_img_matching_list_file.write(result_img_dir[distance_ranking[0][i]])
            tmp_img_matching_list_file.write('\n')
        #
        # ## 14/04/28 here we've done first retrieval of image. But not good...
        tmp_img_matching_list_file.close()
        # raw_input("Press Enter to continue...")

        #### 14/07/25 ranking consistancy measures
        #### First, we consider top several images.
        top_to_check_num = 200
        top_result_first_ranking = []
        RBO_mat = np.zeros((1,top_to_check_num), float)
        for top_img_to_check_i in range(top_to_check_num):
            top_img_ID = first_ranking[top_img_to_check_i]
            TF_IDF_top_tmp =  np.reshape(TF_IDF_norm_matrix[top_img_ID,:],(1,-1))

            L2_distance_tmp = np.zeros((1,image_count), np.float64)

            for image_traversal_i in range(len(result_img_dir)):
                L2_distance_tmp[0,image_traversal_i] = np.dot((TF_IDF_top_tmp.flatten() - TF_IDF_norm_matrix[image_traversal_i,:]),(TF_IDF_top_tmp.flatten() - TF_IDF_norm_matrix[image_traversal_i,:]))

            distance_ranking_tmp = np.argsort(L2_distance_tmp, axis = 1)
            top_result_first_ranking_tmp = list(distance_ranking_tmp[0])
            top_result_first_ranking.append(top_result_first_ranking_tmp)
            RBO_mat[0, top_img_to_check_i] = rank_biased_overlap(first_ranking, top_result_first_ranking_tmp, 0.9, top_to_check_num)


        RBO_reranking_Idx = np.argsort(RBO_mat, axis = 1)[0][::-1]
        RBO_reranking_front = [first_ranking[RBO_reranking_Idx[iii]] for iii in range(top_to_check_num)]
        RBO_reranking_back = list(first_ranking[top_to_check_num::])
        RBO_reranking = list(RBO_reranking_front)
        RBO_reranking.extend(RBO_reranking_back)

        print first_ranking
        print RBO_reranking


        print 'our little test'
        first_ranking_top200 = first_ranking[0:200]
        RBO_reranking_top200 = RBO_reranking[0:200]
        ggg = [RBO_reranking_top200.index(i) for i in first_ranking_top200]
        print ggg
        print easy_merge_sort(ggg,0)[1]
        raw_input('wait wait wait')


        RBO_verified_file = open(RBO_reranking_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_RBO_reranking.txt','w')

        for RBO_reranking_i in range(first_retrieval_num):
            RBO_verified_file.write(result_img_dir[RBO_reranking[RBO_reranking_i]])
            RBO_verified_file.write('\n')
        RBO_verified_file.close()


