__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
from time import clock

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/here/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'
    # Number of clusters: 128 at present
    cluster_number = 8192
    # Using SIFT here
    des_dimension = 128
    first_retrieval_num = 100
    ## store images' dirs
    target_img_dir_list = []
    ## list inside a list
    target_img_matching_img_list = []
    target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
    target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
    for line in target_img_list:
        target_img_dir_list.append(line[:-1])
        # print line[:-1]
    target_img_list.close()
    print len(target_img_dir_list)

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

    # read the TF_IDF_matrix from file.
    # prepare for further VW matching under tf-idf structure
    read_tf_idf_start = clock()
    TF_IDF_append = 'TF_IDF_matrix.txt'
    TF_IDF_dir = top_dir + TF_IDF_append
    TF_IDF_file = open(TF_IDF_dir, 'rb')
    line = TF_IDF_file.readline()
    # TF_IDF_matrix
    TF_IDF_matrix = np.zeros((int(line.split(',')[0]), int(line.split(',')[1])), np.float64)
    for i in range(TF_IDF_matrix.shape[0]):
        line = TF_IDF_file.readline()
        TF_IDF_matrix[i,:] = np.float64(line.split(','))
    ## print TF_IDF_matrix
    TF_IDF_file.close()

    read_tf_idf_end = clock()

    print 'read_tf_idf: ', read_tf_idf_end - read_tf_idf_start



    ##change 14/05/01
    for target_i in range(len(target_img_dir_list)):
        tmp_img_matching_list_file = open(target_img_dir_list[target_i][:-3] + 'txt', 'w')
        tmp_img_matching_img_list = []
        # print target_img_dir_list[target_i]
        ## import target image
        img = cv2.imread(target_img_dir_list[target_i])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT(nfeatures=250)
        kpts_target, des_target = sift.detectAndCompute(img_gray, None)
        target_image_keypoint_num = len(kpts_target)
        print 'kpts numbers of ', target_img_dir_list[target_i], ' : ', len(kpts_target)
        #
        #
        # Allocate each new descriptor to a certain cluster.

        part_1_start = clock()
        target_image_keypoint_labels = np.zeros((1,len(kpts_target)), np.int32)
        for i in range(len(kpts_target)):
            distance_calc = np.zeros((1,centers.shape[0]), np.float32)
            # big_des_target = np.tile(des_target[i],(centers.shape[0],1))
            # (big_des_target - centers)
            for j in range(centers.shape[0]):
                distance_calc[0,j] = np.dot((np.transpose(des_target[i] - centers[j])),(des_target[i] - centers[j]))
            ## argsort? we want to find the min distance, and that would be the nearest cluster center.
            target_image_keypoint_labels[0,i] = distance_calc.argsort(axis = 1)[0, 0]

        part_1_end = clock()
        print 'part_1: ', part_1_end - part_1_start

        ## generate VW for target image.
        part_2_start = clock()
        target_image_VW = np.zeros((1, cluster_number), np.int32)
        for i in range(target_image_keypoint_labels.shape[1]):
            target_image_VW[0, target_image_keypoint_labels[0,i]] += 1

        print target_image_VW
        part_2_end = clock()

        print 'part_2: ', part_2_end - part_2_start




        # ## new image's descriptor file output
        # part_3_start = clock()
        # # target_image_des_dir = target_img_dir_list[target_i][:-4] + '_des.csv'
        #
        # that_file = open(target_img_dir_list[target_i][:-4] + '_des.csv', 'w')
        # for i in range(des_target.shape[0]):
        #     for j in range(des_target.shape[1]):
        #         that_file.write(str(des_target[i, j]))
        #         if j < (des_target.shape[1]-1):
        #             that_file.write(',')
        #     that_file.write('\n')
        # that_file.close()
        # part_3_end = clock()
        #
        # print 'part 3 : ', part_3_end - part_3_start
        #
        ## new image's kpts file output

        # that_file = open(target_img_dir_list[target_i][:-4] + '_kpts.csv', 'w')
        # for i in range(len(kpts_target)):
        #     that_file.write(str(kpts_target[i].pt[0]))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].pt[1]))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].size))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].angle))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].response))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].octave))
        #     that_file.write(str(','))
        #     that_file.write(str(kpts_target[i].class_id))
        #     that_file.write('\n')
        # that_file.close()
        #
        ## new image's VW file output
        # target_image_VW_file = open(target_img_dir_list[target_i][:-4] + '_VW.txt', 'w')
        # target_image_VW_file.write(str(len(kpts_target)))
        # target_image_VW_file.write(',')
        # target_image_VW_file.write(str(cluster_number))
        # target_image_VW_file.write('\n')
        # for i in range(target_image_keypoint_labels.shape[1]):
        #     target_image_VW_file.write(str(target_image_keypoint_labels[0,i]))
        #     if i < (target_image_keypoint_labels.shape[1] - 1):
        #         target_image_VW_file.write(',')
        # target_image_VW_file.write('\n')
        # for i in range(target_image_VW.shape[1]):
        #     target_image_VW_file.write(str(target_image_VW[0,i]))
        #     if i < (target_image_VW.shape[1] - 1):
        #         target_image_VW_file.write(',')
        # target_image_VW_file.write('\n')
        # target_image_VW_file.close()

        # ##########
        #
        # # now we calculate the "distance" between each database image and target image

        result_img_dir =[]
        result_img_kpts = []
        index_file = open('C:/Cassandra/here/image_index_python.txt','rb')
        image_count = 0
        for line in index_file:
            result_img_dir.append((line.split(','))[0])
            result_img_kpts.append(int(float(line.split(',')[1][:-2])))
        # print result_img_dir
        # print result_img_kpts
        index_file.close()
        image_count = len(result_img_dir)
        distance_between_image = np.zeros((1,image_count), np.float64)

        ## Use the right tf-idf Matrix!!!!!!!!  14/04/28
        for i in range(len(result_img_dir)):
            the_file = open((result_img_dir[i].split('.'))[0] + '_VW.txt','r')
            line = the_file.readline()
            line = the_file.readline()
            # read the third line
            line = the_file.readline()
            # get the VW of database image
            VW_tmp = np.array(map(np.int32,line.split(',')))
            # print type(VW_tmp)
            the_file.close()
            # create a eye matrix with tf-idf values.
            TF_IDF_eye = TF_IDF_matrix[i,:]
            aaa = np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)
            # print type(np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye))
            # print np.multiply(aaa, np.float64(target_image_VW - VW_tmp))
            # calculate distance.
            # distance_between_image[0, i] = np.dot((np.dot(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)),
            #                                       np.transpose(np.float64(target_image_VW - VW_tmp)))
            distance_between_image[0, i] = np.dot((np.multiply(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)),
                                                  np.transpose(np.float64(target_image_VW - VW_tmp)))
        distance_ranking = np.argsort(distance_between_image, axis=1)
        # # print distance_ranking
        #
        for i in range(first_retrieval_num):
            # print distance_between_image[0,distance_ranking[0,i]]
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