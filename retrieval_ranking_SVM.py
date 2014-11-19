__author__ = 'LIMU_North'

import cv2
import numpy as np
import os
import math
import subprocess
from sklearn import svm, linear_model
from time import clock



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
    DQE_reranking_dir = top_dir + 'DQE_reranking/'

    Ranking_SVM_file_dir = top_dir + 'Ranking_SVM_file/'
    Ranking_SVM_exe_dir = top_dir + 'Ranking_SVM_exe/'
    Ranking_SVM_reranking_dir = top_dir + 'Ranking_SVM_reranking/'

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
        os.stat(Ranking_SVM_file_dir)
    except:
        os.mkdir(Ranking_SVM_file_dir)
    try:
        os.stat(Ranking_SVM_reranking_dir)
    except:
        os.mkdir(Ranking_SVM_reranking_dir)

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

    ####  start: 2014/09/16 ####
    ####  save the data norm-VWs into .bat needed by ranking-SVM exe
    # TF_IDF_norm_matrix_bat_file = open(Ranking_SVM_exe_dir + 'dataset_TF_IDF_norm.dat','w')
    # for write_i in range(TF_IDF_norm_matrix.shape[0]):
    #     TF_IDF_norm_matrix_bat_file.write('0 qid:1')
    #     for write_j in range(TF_IDF_norm_matrix.shape[1]):
    #         if 0 != TF_IDF_norm_matrix[write_i, write_j]:
    #             TF_IDF_norm_matrix_bat_file.write(' ' + str(write_j+1) + ':' +str(TF_IDF_norm_matrix[write_i, write_j]))
    #     TF_IDF_norm_matrix_bat_file.write('\n')
    # TF_IDF_norm_matrix_bat_file.close()
    #
    # raw_input('output ready. end it.')
    ####  end: 2014/09/16 ####

    retrieval_time_used_start = clock()

    print '...loading finished...'

    ## 07/15 record on how many support vector used.
    num_support_vector_list = []
    ## 14/06/14
    SV_time_list = []
    SV_got_list = []

    DQE_predict_time_list = []

    DQE_train_list = []
    ##change 14/05/01
    for query_i in range(len(query_img_dir_list)):
        print '...starting query ', str(query_i+1), '...'
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

            L2_distance_between_image[0, i] = np.dot((TF_IDF_eye.flatten() - query_TF_IDF_norm.flatten()),(TF_IDF_eye.flatten() - query_TF_IDF_norm.flatten()))
                # L2_distance_between_image[0, i] = np.dot(TF_IDF_eye.flatten(),query_TF_IDF.flatten()) / (math.sqrt(np.dot(TF_IDF_eye.flatten(), TF_IDF_eye.flatten())) * math.sqrt(np.dot(query_TF_IDF.flatten(), query_TF_IDF.flatten())))
            # L2_distance_between_image[0, i] = np.dot((np.multiply(np.float64(query_image_VW - VW_tmp), TF_IDF_eye)),
            #                                       np.transpose(np.float64(query_image_VW - VW_tmp)))
        print len(np.where(L2_distance_between_image == 0)[1])
        distance_ranking = np.argsort(L2_distance_between_image, axis=1)
        ## this ranking need to be inverse first
        print distance_ranking[0]
        first_ranking = list(distance_ranking[0])
        # first_ranking = list(distance_ranking[0][::-1])
        print 'first_ranking: ', first_ranking
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
        ####  Adding spatial verification part into it.
        tmp_SV_result_name = []
        tmp_SV_result_index = []
        ## Re-write the logic. We have the 1st ranking list. Now we take maybe top 200 images into SV.
        SV_IDF_score = []



        tmp_SV_time_start = clock()

        for result_img_i in range(num_for_SV):
            ## First we read their kpts & desc
            ## read desc
            tmp_desc_file = open(database_desc_dir + ranked_result_name_dir[result_img_i] + '_des.csv', 'r')
            tmp_desc_list = []
            for line in tmp_desc_file:
                tmp_desc_list.append(np.float32(line.split(',')))
            desc_tmp = np.array(tmp_desc_list)
            tmp_desc_file.close()

            ## read kpts
            tmp_kpts_file = open(database_kpts_dir + ranked_result_name_dir[result_img_i] + '_kpts.csv', 'r')
            kpts_tmp = []
            for line in tmp_kpts_file:
                tmp_use_kpts = cv2.KeyPoint()
                tmp_use_kpts.pt = (np.float64(line.split(',')[0]), np.float64(line.split(',')[1]))
                tmp_use_kpts.size = np.float64(line.split(',')[2])
                tmp_use_kpts.angle = np.float64(line.split(',')[3])
                tmp_use_kpts.response = np.float64(line.split(',')[4])
                tmp_use_kpts.octave = np.int32(line.split(',')[5])
                tmp_use_kpts.class_id = np.int32(line.split(',')[6])
                kpts_tmp.append(tmp_use_kpts)
            tmp_kpts_file.close()

            ## read the VW? I think we should. Read the normalized one would be better.
            tmp_VW_file = open(database_VW_dir + ranked_result_name_dir[result_img_i] + '_VW.txt', 'r')
            tmp_VW_file.readline()
            ## second line for the assignment of kpts to VW
            line = tmp_VW_file.readline()
            tmp_kpts_to_VW_list = np.int32(line.split(','))
            tmp_kpts_to_VW = np.array(tmp_kpts_to_VW_list)

            ## 3rd line for the normalized VW
            line = tmp_VW_file.readline()
            tmp_norm_VW_list = np.float64(line.split(','))
            tmp_norm_VW = np.array(tmp_norm_VW_list)
            tmp_VW_file.close()

            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = matcher.knnMatch(desc_query, trainDescriptors = desc_tmp, k = 2)
            good_match = []
            trainIdx_range = []
            for j in range(len(raw_matches)):
                trainIdx_range.append(raw_matches[j][0].trainIdx)
                if (raw_matches[j][0].distance / raw_matches[j][1].distance) <= float(pow(nnThreshold,2)) \
                        and raw_matches[j][0].queryIdx < len(kpts_query) and raw_matches[j][0].trainIdx < len(kpts_tmp):
                    good_match.append(raw_matches[j][0])

            if len(good_match) >= minGoodMatch:
                src_pts = np.reshape(np.float32([ kpts_query[m.queryIdx].pt for m in good_match ]),(-1,1,2))
                dst_pts = np.reshape(np.float32([ kpts_tmp[m.trainIdx].pt for m in good_match ]),(-1,1,2))

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                matchesMask = mask.ravel().tolist()

                h,w = img.shape[0],img.shape[1]

                pts = np.reshape(np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]),(-1,1,2))
                dst = cv2.perspectiveTransform(pts,M)

                ## 06/18 We need to change this part. Check the number of inliers

                if np.array(matchesMask).sum() >= 10 :
                    ## too early
                    tmp_SV_result_name.append(ranked_result_name_dir[result_img_i])
                    tmp_SV_result_index.append(first_ranking[result_img_i])

                    ## addition for 2nd ranking.
                    #### 2014/09/16 seems that this part have some major change, that makes it better. Be cautious.
                    tmp_IDF_score = 0
                    tmp_VW_appeared = []

                    matching_point = np.array(matchesMask)
                    matching_point_Idx = np.where(matching_point == 1)[0].tolist()

                    for good_match_i in range(len(matching_point_Idx)):
                        tmp_VW_appeared.append(query_image_keypoint_labels[good_match[matching_point_Idx[good_match_i]].queryIdx])
                    tmp_VW_appeared = list(set(tmp_VW_appeared))
                    # print tmp_VW_appeared

                    tmp_VW_appeared_mat = np.zeros((cluster_number,), np.float32)

                    for good_match_i in range(len(good_match)):
                        tmp_VW_appeared_mat[query_image_keypoint_labels[good_match[good_match_i].queryIdx]] = 1.0

                    tmp_IDF_score = np.dot(IDF_matrix.flatten(), tmp_VW_appeared_mat)
                    # for appeared_VW_i in range(len(tmp_VW_appeared)):
                    #     tmp_IDF_score += IDF_matrix[0,tmp_VW_appeared[appeared_VW_i]] * tmp_norm_VW[appeared_VW_i]
                    SV_IDF_score.append(tmp_IDF_score)
                else:
                    # print 'Not enough inliers in RANSAC... Abandoned...'
                    matchesMask = None
            else:
                # print "Not enough matches are found - %d/%d" % (len(good_match),minGoodMatch)
                matchesMask = None
        print 'number of Verified images: ', len(SV_IDF_score)
        SV_got_list.append(len(SV_IDF_score))

        ## Now wo do the 2nd reranking.

        ## here's the problem
        ## first_ranking is alright
        # without_SV_Idx = list(first_ranking - tmp_SV_result_index)
        without_SV_Idx = list(first_ranking[num_for_SV:])
        #### improve version
        SV_failed_Idx = list(first_ranking[0:num_for_SV])
        if len(tmp_SV_result_index) > 0:
            for element_x in tmp_SV_result_index:
                SV_failed_Idx.remove(element_x)
        ####
        ranking_SV_Idx = []
        SV_reranking_Idx_list = []
        # raw_input("Press Enter to continue...")
        if len(SV_IDF_score) > 0:
            ## 2014/06/25 night
            SV_IDF_ranking = np.argsort(SV_IDF_score)[::-1]
            for ranking_SV_i in range(SV_IDF_ranking.shape[0]):
                ranking_SV_Idx.append(tmp_SV_result_index[SV_IDF_ranking[ranking_SV_i]])
            SV_reranking_Idx_list = list(ranking_SV_Idx)
            SV_reranking_Idx_list.extend(SV_failed_Idx)
            SV_reranking_Idx_list.extend(without_SV_Idx)
        else:
            ## be careful about list(set())
            SV_reranking_Idx_list.extend(SV_failed_Idx)
            SV_reranking_Idx_list_ = list(without_SV_Idx)
        ####
        print 'verified image list: ', ranking_SV_Idx
        print len(ranking_SV_Idx)
        print 'SV_reranking_Idx_list: ', SV_reranking_Idx_list
        print len(SV_reranking_Idx_list)

        #### write record of SV result.
        print
        print '...number of spatially verified images: ', len(tmp_SV_result_name), '...'
        print
        tmp_SV_time_end = clock()
        print '...time used for SV for query number %d: %d minutes %f seconds...' %(query_i,np.int32((tmp_SV_time_end - tmp_SV_time_start)/60), (tmp_SV_time_end - tmp_SV_time_start)%60)
        SV_time_list.append(tmp_SV_time_end - tmp_SV_time_start)
        SV_verified_file = open(SV_result_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_SV_result.txt','w')
        SV_verified_file.write(str(len(tmp_SV_result_name)))
        SV_verified_file.write('\n')
        if len(tmp_SV_result_name) > 0:
            for SV_result_i in range(len(tmp_SV_result_name)):
                SV_verified_file.write(database_image_dir + tmp_SV_result_name[SV_result_i] + '.jpg')
                SV_verified_file.write('\n')
        SV_verified_file.close()

        SV_reranking_file = open(SV_reranking_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_SV_reranking.txt','w')
        for SV_reranking_i in range(first_retrieval_num):
            SV_reranking_file.write(result_img_dir[SV_reranking_Idx_list[SV_reranking_i]])
            SV_reranking_file.write('\n')
        SV_reranking_file.close()

        #### Generate a string series for describing ranking
        ranking_score_appeared = list(set(SV_IDF_score))

        #### sort in acending sequence
        ranking_score_appeared.sort()

        ranking_SV_score = []
        ranking_SV_Idx_ranked = []
        SV_IDF_ranking = np.argsort(SV_IDF_score)[::-1]

        for ranking_SV_i in range(SV_IDF_ranking.shape[0]):
            ranking_SV_score.append(SV_IDF_score[SV_IDF_ranking[ranking_SV_i]])
            ranking_SV_Idx_ranked.append(SV_reranking_Idx_list[SV_IDF_ranking[ranking_SV_i]])
        ####

        SV_for_ranking_SVM_file = open(Ranking_SVM_file_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_Ranking_SVM_file.dat','w')


        #### retrieve the norm-VW of each image verified.
        # SV_for_ranking_SVM_file.write('## This is a test for ranking SVM input.\n')

        #### 14/11/07 test on 50/50
        numPositivesMax = 50
        numNegatives = 200

        positiveTaken = min(numPositivesMax, SV_IDF_ranking.shape[0])

        #### 1. Good examples.
        for ranking_SVM_i in range(positiveTaken):
        # for ranking_SVM_i in range(SV_IDF_ranking.shape[0]):
            #### First, output the ranking grade for this image.
            SV_for_ranking_SVM_file.write(str(ranking_score_appeared.index(SV_IDF_score[ranking_SVM_i]) + 1))
            SV_for_ranking_SVM_file.write(' qid:1')
            #### ... then, output the norm VW.
            for ranking_SVM_VW_j in range(cluster_number):
                # print TF_IDF_norm_matrix[ranking_SV_Idx_ranked[ranking_SVM_i],ranking_SVM_VW_j]
                if 0 != TF_IDF_norm_matrix[ranking_SV_Idx_ranked[ranking_SVM_i],ranking_SVM_VW_j]:
                    SV_for_ranking_SVM_file.write(' ' + str(ranking_SVM_VW_j+1) + ':')
                    SV_for_ranking_SVM_file.write(str(TF_IDF_norm_matrix[ranking_SV_Idx_ranked[ranking_SVM_i],ranking_SVM_VW_j]))
            SV_for_ranking_SVM_file.write('\n')

        #### 2. Bad examples.
        for ranking_SVM_i in range(numNegatives):
            SV_for_ranking_SVM_file.write(str(0))
            SV_for_ranking_SVM_file.write(' qid:1')
            for ranking_SVM_VW_j in range(cluster_number):
                # print TF_IDF_norm_matrix[ranking_SV_Idx_ranked[ranking_SVM_i],ranking_SVM_VW_j]
                if 0 != TF_IDF_norm_matrix[first_ranking[-ranking_SVM_i-1],ranking_SVM_VW_j]:
                    SV_for_ranking_SVM_file.write(' ' + str(ranking_SVM_VW_j+1) + ':')
                    SV_for_ranking_SVM_file.write(str(TF_IDF_norm_matrix[first_ranking[-ranking_SVM_i-1],ranking_SVM_VW_j]))
            SV_for_ranking_SVM_file.write('\n')

        SV_for_ranking_SVM_file.close()

        #### 2014/09/16 ####
        #### using ranking-svm exe from outside. use subprocess.
        train_dat_dir = Ranking_SVM_file_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_Ranking_SVM_file.dat'
        #### training
        train_returnCode  = subprocess.call(Ranking_SVM_exe_dir + 'svm_rank_learn.exe' + ' -c 0.0025 '+ train_dat_dir + ' ' + Ranking_SVM_exe_dir +'model.dat')
        # print train_returnCode

        #### testing
        test_returnCode = subprocess.call(Ranking_SVM_exe_dir + 'svm_rank_classify.exe' + ' '+ Ranking_SVM_exe_dir + 'dataset_TF_IDF_norm.dat ' + Ranking_SVM_exe_dir + 'model.dat ' +Ranking_SVM_exe_dir + 'prediction.dat')
        # print test_returnCode


        #### read result from prediction.dat
        prediction_file = open(Ranking_SVM_exe_dir + 'prediction.dat', 'r')
        prediction_score = []

        for line in prediction_file:
            prediction_score.append(float(line))
        prediction_file.close()
        prediction_score = np.array(prediction_score)

        #### descending order on score
        Ranking_SVM_rerank = np.argsort(prediction_score)[::-1]
        # print prediction_score
        # print Ranking_SVM_rerank


        # raw_input('waiting for command...')

        #### Final output of reranking
        Ranking_SVM_reranking_file = open(Ranking_SVM_reranking_dir + ((query_img_dir_list[query_i].split('/'))[-1]).split('.')[0] + '_Ranking_SVM.txt','w')
        for reranking_i in range(first_retrieval_num):
            Ranking_SVM_reranking_file.write(result_img_dir[Ranking_SVM_rerank[reranking_i]])
            Ranking_SVM_reranking_file.write('\n')
        Ranking_SVM_reranking_file.close()

        # raw_input('output ready...')
        #### 2014/09/16 ####
        print '...query ', str(query_i + 1), ' finished...'


    # raw_input("Press Enter to continue...")
    retrieval_time_used_end = clock()
    print 'Retrieval time used total: ', int((retrieval_time_used_end - retrieval_time_used_start)/60), ' minutes ', \
        (retrieval_time_used_end - retrieval_time_used_start)%60, ' seconds...'

    SV_time_array = np.array(SV_time_list)
    print 'SV time used average: %f seconds' % np.average(SV_time_array)
    print SV_got_list
    print 'average SV image got: ',np.average(np.array(SV_got_list))

    ## extra SV result output 2014/11/19
    SV_result_num_file = open(top_dir + 'SV_result_num'+'_Ranking_SVM.csv','w')
    for SV_i in range(len(SV_got_list)):
        SV_result_num_file.write(str(SV_got_list[SV_i])+'\n')
    SV_result_num_file.close()

