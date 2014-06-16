__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
import os
from time import clock

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/python_oxford/'
    database_image_dir = top_dir + 'database/'
    database_desc_dir = top_dir + 'database_desc/'
    database_kpts_dir = top_dir + 'database_kpts/'
    database_VW_dir = top_dir + 'database_VW/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'
    SV_result_dir = top_dir + 'SV_verified/'
    SV_reranking_dir = top_dir + 'SV_reranking/'

    try:
        os.stat(SV_result_dir)
    except:
        os.mkdir(SV_result_dir)
    try:
        os.stat(SV_reranking_dir)
    except:
        os.mkdir(SV_reranking_dir)

    ## operation bools
    bool_using_tf_idf = True
    bool_read_tf_idf_from_txt = False
    bool_read_idf_from_txt = False
    bool_read_database_VW_from_txt = False

    # Number of clusters: 128 at present
    cluster_number = 8192
    first_retrieval_num = 5062
    # Using SIFT here
    des_dimension = 128
    kpts_density = 238
    # For SV
    num_for_SV = 200
    nnThreshold = 0.8
    minGoodMatch = 10
    ## store images' dirs
    target_img_dir_list = []
    ## list inside a list
    target_img_matching_img_list = []
    # target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
    target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
    for line in target_img_list:
        target_img_dir_list.append(line[:-1])
        # print line[:-1]
    target_img_list.close()
    print 'len(target_img_dir_list): ', len(target_img_dir_list)

    ## read kmeans centers
    kmeans_result_append = '/kmeans_result.txt'
    kmeans_result_dir = top_dir + kmeans_result_append
    kmeans_result_file = open(kmeans_result_dir, 'rb')

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
        database_VW_matrix = np.load(top_dir + 'database_VW_matrix.npy')


    print '...database image VWs loaded...'
    print

    retrieval_time_used_start = clock()

    ## 14/06/14
    SV_time_list = []

    ##change 14/05/01
    for target_i in range(len(target_img_dir_list)):
        tmp_img_matching_list_file = open(target_img_dir_list[target_i][:-3] + 'txt', 'w')
        tmp_img_matching_img_list = []
        # print target_img_dir_list[target_i]
        ## import target image
        img = cv2.imread(target_img_dir_list[target_i])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        num_feature = int(img_gray.shape[0] * img_gray.shape[1] / 357)
        sift = cv2.SIFT(nfeatures=num_feature, edgeThreshold=0.01)
        kpts_target, desc_target = sift.detectAndCompute(img_gray, None)
        target_image_keypoint_num = len(kpts_target)
        print 'kpts numbers of ', target_img_dir_list[target_i], ' : ', len(kpts_target)
        #
        #
        # Allocate each new descriptor to a certain cluster.

        part_1_start = clock()
        target_image_keypoint_labels = np.zeros((1,len(kpts_target)), np.int32)
        for i in range(len(kpts_target)):
            distance_calc = np.zeros((1,centers.shape[0]), np.float32)
            # big_desc_target = np.tile(desc_target[i],(centers.shape[0],1))
            # (big_desc_target - centers)
            for j in range(centers.shape[0]):
                distance_calc[0,j] = np.dot((np.transpose(desc_target[i] - centers[j])),(desc_target[i] - centers[j]))
            ## argsort? we want to find the min distance, and that would be the nearest cluster center.
            target_image_keypoint_labels[0,i] = distance_calc.argsort(axis = 1)[0, 0]
        # print 'target_image_keypoint_labels: ', target_image_keypoint_labels
        part_1_end = clock()
        print '...part 1 time: ', part_1_end - part_1_start, ' seconds...'

        ## generate VW for target image.
        part_2_start = clock()
        target_image_VW = np.zeros((1, cluster_number), np.int32)
        for i in range(target_image_keypoint_labels.shape[1]):
            target_image_VW[0, target_image_keypoint_labels[0,i]] += 1
        print 'target_image_VW: ', target_image_VW[0,:]
        target_image_VW = np.float64(target_image_VW)/np.float64(target_image_VW.sum(axis=1)[0])
        part_2_end = clock()

        print '...part 2 time: ', part_2_end - part_2_start, ' seconds...'

        # ## new image's descriptor file output
        part_3_start = clock()
        # target_image_des_dir = target_img_dir_list[target_i][:-4] + '_des.csv'

        that_file = open(target_img_dir_list[target_i][:-4] + '_des.csv', 'w')
        for i in range(desc_target.shape[0]):
            for j in range(desc_target.shape[1]):
                that_file.write(str(desc_target[i, j]))
                if j < (desc_target.shape[1]-1):
                    that_file.write(',')
            that_file.write('\n')
        that_file.close()
        part_3_end = clock()

        print '...part 3 time: ', part_3_end - part_3_start, ' seconds...'

        # new image's kpts file output

        that_file = open(target_img_dir_list[target_i][:-4] + '_kpts.csv', 'w')
        for i in range(len(kpts_target)):
            that_file.write(str(kpts_target[i].pt[0]))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].pt[1]))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].size))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].angle))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].response))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].octave))
            that_file.write(str(','))
            that_file.write(str(kpts_target[i].class_id))
            that_file.write('\n')
        that_file.close()
        #
        ## new image's VW file output
        target_image_VW_file = open(target_img_dir_list[target_i][:-4] + '_VW.txt', 'w')
        target_image_VW_file.write(str(len(kpts_target)))
        target_image_VW_file.write(',')
        target_image_VW_file.write(str(cluster_number))
        target_image_VW_file.write('\n')
        for i in range(target_image_keypoint_labels.shape[1]):
            target_image_VW_file.write(str(target_image_keypoint_labels[0,i]))
            if i < (target_image_keypoint_labels.shape[1] - 1):
                target_image_VW_file.write(',')
        target_image_VW_file.write('\n')
        for i in range(target_image_VW.shape[1]):
            target_image_VW_file.write(str(target_image_VW[0,i]))
            if i < (target_image_VW.shape[1] - 1):
                target_image_VW_file.write(',')
        target_image_VW_file.write('\n')
        target_image_VW_file.close()

        # ##########
        #
        # # now we calculate the "distance" between each database image and target image

        # result_img_dir =[]
        # result_img_kpts = []
        # index_file = open(top_dir + 'image_index_python.txt','rb')
        # image_count = 0
        # for line in index_file:
        #     result_img_dir.append((line.split(','))[0])
        #     result_img_kpts.append(int(float(line.split(',')[1][:-2])))
        # # print result_img_dir
        # # print result_img_kpts
        # index_file.close()
        # image_count = len(result_img_dir)
        distance_between_image = np.zeros((1,image_count), np.float64)
        # target_image_VW_norm = target_image_VW / target_image_VW.sum()
        ## Use the right tf-idf Matrix!!!!!!!!  14/04/28
        for i in range(len(result_img_dir)):
            # the_file = open(database_VW_dir + ((result_img_dir[i].split('/'))[-1]).split('.')[0] + '_VW.txt','r')
            # line = the_file.readline()
            # line = the_file.readline()
            # # read the third line
            # line = the_file.readline()
            # # get the VW of database image
            # VW_tmp = np.array(map(np.float64,line.split(',')))
            # # print type(VW_tmp)
            # the_file.close()
            # create a eye matrix with tf-idf values.
            TF_IDF_eye = np.reshape(TF_IDF_matrix[i,:],(1,-1))
            # TF_IDF_eye = TF_IDF_matrix[i,:]
            # aaa = np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)
            # print type(np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye))
            # print np.multiply(aaa, np.float64(target_image_VW - VW_tmp))
            # calculate distance.
            # distance_between_image[0, i] = np.dot((np.dot(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)),
            #                                       np.transpose(np.float64(target_image_VW - VW_tmp)))

            ##14/05/05  normalize the VW, and then calculate distance.

            # VW_tmp_norm = VW_tmp / VW_tmp.sum()
            if bool_using_tf_idf:
                distance_between_image[0, i] = np.dot((np.multiply(np.float64(target_image_VW - database_VW_matrix[i,:]), TF_IDF_eye)),
                                                   np.transpose(np.multiply(np.float64(target_image_VW - database_VW_matrix[i,:]), TF_IDF_eye)))
            else:
                distance_between_image[0, i] = np.dot((np.float64(target_image_VW - database_VW_matrix[i,:])), np.transpose(np.float64(target_image_VW - database_VW_matrix[i,:])))

            # distance_between_image[0, i] = np.dot((np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)),
            #                                       np.transpose(np.float64(target_image_VW - VW_tmp)))
        distance_ranking = np.argsort(distance_between_image, axis=1)
        first_ranking = list(distance_ranking[0])
        print 'first_ranking: ', first_ranking
        ranked_result_name_dir = []

        for i in range(first_retrieval_num):
            # print distance_between_image[0,distance_ranking[0,i]]
            ranked_result_name_dir.append((result_img_dir[distance_ranking[0][i]].split('.')[0]).split('/')[-1])
            tmp_img_matching_list_file.write(result_img_dir[distance_ranking[0][i]])
            tmp_img_matching_list_file.write('\n')
            # img_tmp = cv2.imread(result_img_dir[distance_ranking[0][i]],0 )
            # img_tmp = cv2.resize(img_tmp, (0,0), fx=0.5, fy=0.5)
            # cv2.namedWindow(result_img_dir[distance_ranking[0][i]])
            # cv2.imshow(result_img_dir[distance_ranking[0][i]], img_tmp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        #
        # ## 14/04/28 here we've done first retrieval of image. But not good...
        tmp_img_matching_list_file.close()

        ####  Adding spatial verification part into it.
        tmp_SV_result_name = []
        tmp_SV_result_index = []
        ## Re-write the logic. We have the 1st ranking list. Now we take maybe top 200 images into SV.
        SV_IDF_score = []
        tmp_SV_time_start = clock()
        for result_img_i in range(num_for_SV):

            tmp_SV_img = cv2.imread(database_image_dir + ranked_result_name_dir[result_img_i] + '.jpg',0)


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
            raw_matches = matcher.knnMatch(desc_target, trainDescriptors = desc_tmp, k = 2)
            good_match = []

            trainIdx_range = []
            for j in range(len(raw_matches)):
                trainIdx_range.append(raw_matches[j][0].trainIdx)
                if (raw_matches[j][0].distance / raw_matches[j][1].distance) <= float(pow(nnThreshold,2)) \
                        and raw_matches[j][0].queryIdx < len(kpts_target) and raw_matches[j][0].trainIdx < len(kpts_target):
                    good_match.append(raw_matches[j][0])
            ###
            cv2.rectangle(tmp_SV_img,(0,0),(tmp_SV_img.shape[1],tmp_SV_img.shape[0]),(0,255,0),thickness=20)
            ###

            if len(good_match) >= minGoodMatch:
                # if target_i == 32:
                #     print len(kpts_target)
                #     print len(kpts_tmp)
                #     raw_input("Press Enter to continue...")
                src_pts = np.reshape(np.float32([ kpts_target[m.queryIdx].pt for m in good_match ]),(-1,1,2))
                dst_pts = np.reshape(np.float32([ kpts_tmp[m.trainIdx].pt for m in good_match ]),(-1,1,2))

                homograph_start = clock()
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                homograph_end = clock()
                # print 'Homograph time used: ', homograph_end - homograph_start
                matchesMask = mask.ravel().tolist()

                h,w = img.shape[0],img.shape[1]

                # print h,w
                pts = np.reshape(np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]),(-1,1,2))
                dst = cv2.perspectiveTransform(pts,M)
                # print M
                # print dst
                cv2.polylines(tmp_SV_img,[np.int32(dst)],True,(0,0,255),5, 1)
                tmp_SV_result_name.append(ranked_result_name_dir[result_img_i])
                tmp_SV_result_index.append(first_ranking[result_img_i])

                ## addition for 2nd ranking.
                tmp_IDF_score = 0
                tmp_VW_appeared = []
                for good_match_i in range(len(good_match)):
                    tmp_VW_appeared.append(target_image_keypoint_labels[0,good_match[good_match_i].queryIdx])
                    # tmp_VW_belonging = target_image_keypoint_labels[0,good_match[good_match_i].queryIdx]
                    # tmp_IDF_score += IDF_matrix[0,tmp_VW_belonging]

                tmp_VW_appeared = list(set(tmp_VW_appeared))
                for appeared_VW_i in range(len(tmp_VW_appeared)):
                    tmp_IDF_score += IDF_matrix[0,tmp_VW_appeared[appeared_VW_i]] * tmp_norm_VW[appeared_VW_i]
                SV_IDF_score.append(tmp_IDF_score)
            else:
                # print "Not enough matches are found - %d/%d" % (len(good_match),minGoodMatch)
                matchesMask = None
        print 'len(SV_IDF_score): ', len(SV_IDF_score)
        ## Now wo do the 2nd reranking.

        ## here's the problem
        ## first_ranking is alright
        # without_SV_Idx = list(first_ranking - tmp_SV_result_index)
        without_SV_Idx = list(first_ranking)
        if len(tmp_SV_result_index) > 0:
            for element_x in tmp_SV_result_index:
                without_SV_Idx.remove(element_x)
        print 'without_SV_Idx: ', without_SV_Idx
        print len(without_SV_Idx)

        ## this below return a 1-D array
        ranking_SV_Idx = []
        SV_reranking_Idx_list = []
        print 'tmp_SV_result_index: ', tmp_SV_result_index
        # raw_input("Press Enter to continue...")
        if len(SV_IDF_score) > 0:
            ## SV_reranking_Idx_list is empty
            SV_IDF_ranking = np.argsort(SV_IDF_score)
            for ranking_SV_i in range(SV_IDF_ranking.shape[0]):
                # print result_img_dir[tmp_SV_result_index[SV_IDF_ranking[ranking_SV_i]]]
                ranking_SV_Idx.append(tmp_SV_result_index[SV_IDF_ranking[ranking_SV_i]])
            # print 'SV_reranking_Idx_list nothing: ', SV_reranking_Idx_list
            # raw_input('1')
            ## push SV result into SV_reranking_Idx_list
            SV_reranking_Idx_list = list(ranking_SV_Idx)
            # print 'SV_reranking_Idx_list 2 :', SV_reranking_Idx_list
            # raw_input('2')
            ## push rest images into SV_reranking_Idx_list
            SV_reranking_Idx_list.extend(without_SV_Idx)
            # print 'SV_reranking_Idx_list 3: ', SV_reranking_Idx_list
            # raw_input('3')
        else:
            ## be careful about list(set())
            SV_reranking_Idx_list = list(without_SV_Idx)
        print 'SV_reranking_Idx_list: ', SV_reranking_Idx_list
        print len(SV_reranking_Idx_list)
        # raw_input("Press Enter to continue...")
        #### write record of SV result.
        print
        print '...number of spatially verified images: ', len(tmp_SV_result_name), '...'
        print
        tmp_SV_time_end = clock()
        print '...time used for SV for query number %d: %d minutes %f seconds...' %(target_i,np.int32((tmp_SV_time_end - tmp_SV_time_start)/60), (tmp_SV_time_end - tmp_SV_time_start)%60)
        SV_time_list.append(tmp_SV_time_end - tmp_SV_time_start)
        SV_verified_file = open(SV_result_dir + ((target_img_dir_list[target_i].split('/'))[-1]).split('.')[0] + '_SV_result.txt','w')
        SV_verified_file.write(str(len(tmp_SV_result_name)))
        SV_verified_file.write('\n')
        if len(tmp_SV_result_name) > 0:
            for SV_result_i in range(len(tmp_SV_result_name)):
                SV_verified_file.write(database_image_dir + tmp_SV_result_name[SV_result_i] + '.jpg')
                SV_verified_file.write('\n')
        SV_verified_file.close()

        SV_reranking_file = open(SV_reranking_dir + ((target_img_dir_list[target_i].split('/'))[-1]).split('.')[0] + '_SV_reranking.txt','w')

        for SV_reranking_i in range(first_retrieval_num):
            SV_reranking_file.write(result_img_dir[SV_reranking_Idx_list[SV_reranking_i]])
            SV_reranking_file.write('\n')
        SV_reranking_file.close()

    # raw_input("Press Enter to continue...")
    retrieval_time_used_end = clock()
    print 'Retrieval time used total: ', int((retrieval_time_used_end - retrieval_time_used_start)/60), ' minutes ', \
        (retrieval_time_used_end - retrieval_time_used_start)%60, ' seconds...'

    SV_time_array = np.array(SV_time_list)
    print 'SV time used average: %f seconds' % np.average(SV_time_array)

