__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
import os
from time import clock

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/python_oxford/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'
    dataset_dir = 'C:/Cassandra/oxbuild_images/'
    SV_output_dir = 'C:/Cassandra/SV_output/'

    try:
        os.stat(SV_output_dir)
    except:
        os.mkdir(SV_output_dir)

    #########
    top_result_taken = 5
    query_wanna_check = 'ashmolean_2_'
    ##############
    tmp_good_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + query_wanna_check + 'good.txt'
    tmp_ok_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + query_wanna_check + 'ok.txt'
    tmp_junk_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + query_wanna_check + 'junk.txt'
    tmp_good_file = open(tmp_good_dir,'r')
    tmp_ok_file = open(tmp_ok_dir,'r')
    tmp_junk_file = open(tmp_junk_dir,'r')
    tmp_good = []
    tmp_ok = []
    tmp_junk = []
    # tmp_result = []
    good_count_tmp = 0
    ok_count_tmp = 0
    junk_count_tmp = 0
    negative_count_tmp = 0

    for line in tmp_good_file:
        tmp_good.append(line[:-1])
    for line in tmp_ok_file:
        tmp_ok.append(line[:-1])
    for line in tmp_junk_file:
        tmp_junk.append(line[:-1])
    tmp_good_file.close()
    tmp_ok_file.close()
    tmp_junk_file.close()
    print tmp_good
    print tmp_ok


    top_several_results = []
    top_results_judge = np.zeros((1,top_result_taken), np.int32)
    result_file = open(query_goto_dir + query_wanna_check + 'query.txt','r')
    for i in range(top_result_taken):
        line = result_file.readline()
        print (line.split('/')[-1]).split('.')[0]

        top_several_results.append((line.split('/')[-1]).split('.')[0])
        is_negative = True
        for j in range(len(tmp_good)):
            if tmp_good[j].find((line.split('/')[-1]).split('.')[0]) == 0:
                print 'GOOD!'
                top_results_judge[0,i] = 1
                is_negative = False
        for j in range(len(tmp_ok)):
            if tmp_ok[j].find((line.split('/')[-1]).split('.')[0]) == 0:
                top_results_judge[0,i] = 1
                is_negative = False
        for j in range(len(tmp_junk)):
            if tmp_junk[j].find((line.split('/')[-1]).split('.')[0]) == 0:
                top_results_judge[0,i] = 0
                is_negative = False
        if is_negative:
            top_results_judge[0,i] = -1
    result_file.close()
    print top_several_results
    print top_results_judge
    result_img = []
    ## read two img's kpts and desc
    # kpts_query_file = open(query_goto_dir + query_wanna_check + 'query_kpts.csv','r')
    # for line in kpts_query_file:

    # kpts_query = open(top_dir + 'database_kpts/' + )


    ## open the query image
    img_query = cv2.imread(query_goto_dir + query_wanna_check + 'query.jpg')
    cv2.imwrite(SV_output_dir + query_wanna_check + 'query.jpg', img_query)
    cv2.imshow('Query', img_query)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sift = cv2.SIFT(nfeatures = (img_query.shape[0]*img_query.shape[1]/238), edgeThreshold=0.01)
    kpts_query, desc_query = sift.detectAndCompute(img_query, None)
    print 'len kpts_query: ', len(kpts_query)
    # print desc_query[0]




    for i in range(len(top_several_results)):
        tmp_img = cv2.imread(dataset_dir + top_several_results[i] + '.jpg')
        # cv2.imshow('break', tmp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        if top_results_judge[0,i] == 1:



            sift = cv2.SIFT(nfeatures = (tmp_img.shape[0]*tmp_img.shape[1]/238), edgeThreshold=0.01)
            kpts_tmp, desc_tmp = sift.detectAndCompute(tmp_img, None)

            print 'len(kpts_tmp): ', len(kpts_tmp)
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = matcher.knnMatch(desc_query, trainDescriptors = desc_tmp, k = 2)
            print 'raw match: ', len(raw_matches)

            nnThreshold = 0.8
            good_match = []
            for j in range(len(raw_matches)):
                if (raw_matches[j][0].distance / raw_matches[j][1].distance) <= float(pow(nnThreshold,2)):
                    good_match.append(raw_matches[j][0])
            # print good_match[0].distance
            print 'good match: ', len(good_match)
            minGoodMatch = 10
            cv2.rectangle(tmp_img,(0,0),(tmp_img.shape[1],tmp_img.shape[0]),(0,255,0),thickness=5)
            if len(good_match) >= minGoodMatch:
                src_pts = np.reshape(np.float32([ kpts_query[m.queryIdx].pt for m in good_match ]),(-1,1,2))
                dst_pts = np.reshape(np.float32([ kpts_tmp[m.trainIdx].pt for m in good_match ]),(-1,1,2))
                print 'src: ', src_pts.shape
                print 'dst: ',dst_pts.shape
                print src_pts

                homograph_start = clock()
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                homograph_end = clock()
                print 'Homograph time used: ', homograph_end - homograph_start
                matchesMask = mask.ravel().tolist()

                h,w = img_query.shape[0],img_query.shape[1]

                # print h,w
                pts = np.reshape(np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]),(-1,1,2))
                dst = cv2.perspectiveTransform(pts,M)
                # print M
                # print dst
                cv2.polylines(tmp_img,[np.int32(dst)],True,(0,0,255),5, 1)
            else:
                print "Not enough matches are found - %d/%d" % (len(good_match),minGoodMatch)
                matchesMask = None


        elif top_results_judge[0,i] == 0:
            # cv2.rectangle(tmp_img,(0,0),(tmp_img.shape[1],tmp_img.shape[0]),(255,255,0),thickness=3)
            pass
        else:
            cv2.rectangle(tmp_img,(0,0),(tmp_img.shape[1],tmp_img.shape[0]),(255,0,0),thickness=5)

        cv2.imwrite(SV_output_dir + 'SV_result_'+ str(i+1) + '_' + top_several_results[i] + '.jpg', tmp_img)
        cv2.imshow(top_several_results[i], tmp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




