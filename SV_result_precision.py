__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
from computeInversion import Kendall_tau
from time import clock

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/python_oxford/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'

    SV_ranking_result_dir = top_dir + 'SV_reranking/'
    SV_verified_dir = top_dir + 'SV_verified/'
    top_retrieval_num = 5062
    num_for_SV = 200
    target_img_dir_list = []
    target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
    for line in target_img_list:
        target_img_dir_list.append(line[:-1].split('/')[-1])
    target_img_list.close()
    print 'Number of query images: ', len(target_img_dir_list)

    total_image_retrieved = []
    good_count = []
    ok_count = []
    junk_count = []
    negative_count = []
    positive_total = []
    output_file = open(top_dir + 'SV_result_precision.csv', 'w')
    new_output_file = open(top_dir + 'SV_200_precision.csv', 'w')
    SV_result_number = []


    ## zeros and ones
    ## read result txt for each query
    for query_i in range(len(target_img_dir_list)):
        tmp_good_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'good.txt'
        tmp_ok_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'ok.txt'
        tmp_junk_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'junk.txt'
        tmp_good_file = open(tmp_good_dir)
        tmp_ok_file = open(tmp_ok_dir)
        tmp_junk_file = open(tmp_junk_dir)
        tmp_good = []
        tmp_ok = []
        tmp_junk = []
        tmp_result = []
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

        #### for calculate tau
        positive_negative_lst = []


        ## count the positives
        # positive_total.append(len(tmp_good)+ len(tmp_ok))
        tmp_result_file = open(SV_verified_dir + target_img_dir_list[query_i].split('.')[0] + '_SV_result.txt', 'r')
        line = tmp_result_file.readline()
        tmp_SV_result_num = int(line.split('\n')[0])
        SV_result_number.append(tmp_SV_result_num)
        tmp_result_file.close()

        tmp_result_file = open(SV_ranking_result_dir + target_img_dir_list[query_i].split('.')[0] + '_SV_reranking.txt', 'r')

        for line_i in range(tmp_SV_result_num):
            line = tmp_result_file.readline()
            tmp_result.append(line.split('/')[:-1])
            # print line.split('/')[:-1]
            is_negative = True
            for i in range(len(tmp_good)):
                if tmp_good[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    is_negative = False
                    good_count_tmp += 1
                    positive_negative_lst.append(1.0)
            for i in range(len(tmp_ok)):
                if tmp_ok[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    is_negative = False
                    ok_count_tmp += 1
                    positive_negative_lst.append(1.0)
            for i in range(len(tmp_junk)):
                if tmp_junk[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    is_negative = False
                    junk_count_tmp += 1
            if is_negative:
                negative_count_tmp += 1
                positive_negative_lst.append(0)
        tmp_precision = (good_count_tmp+ok_count_tmp)/float(good_count_tmp+ok_count_tmp+negative_count_tmp)
        print 'Good: ', good_count_tmp
        print 'OK: ', ok_count_tmp
        print 'Precision: ', tmp_precision

        output_file.write(str(len(tmp_result)))
        output_file.write(',')
        output_file.write(str(good_count_tmp+ok_count_tmp))
        output_file.write(',')
        output_file.write(str(tmp_precision))
        output_file.write(',')
        output_file.write(str(len(tmp_good)+len(tmp_ok)))
        output_file.write(',')
        output_file.write(str(Kendall_tau(positive_negative_lst)))
        output_file.write('\n')
        tmp_result_file.close()

        # if 29 == query_i:
        #     print target_img_dir_list[query_i]
        #     print positive_negative_lst
        #     print Kendall_tau(positive_negative_lst)
        #     raw_input('asdf')


    for query_i in range(len(target_img_dir_list)):
            tmp_good_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'good.txt'
            tmp_ok_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'ok.txt'
            tmp_junk_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('.')[0])[:-5] + 'junk.txt'
            tmp_good_file = open(tmp_good_dir)
            tmp_ok_file = open(tmp_ok_dir)
            tmp_junk_file = open(tmp_junk_dir)
            tmp_good = []
            tmp_ok = []
            tmp_junk = []
            tmp_result = []
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

            #### for calculate tau
            positive_negative_lst = []
            #### open positive or not file
            tmp_file = open(SV_ranking_result_dir + target_img_dir_list[query_i].split('.')[0] + '_SV_reranking.txt', 'r')
            for count_i in range(num_for_SV):
                line = tmp_file.readline()
                tmp_result.append(line.split('/')[:-1])
                # print line.split('/')[:-1]
                is_negative = True
                for i in range(len(tmp_good)):
                    if tmp_good[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                        is_negative = False
                        good_count_tmp += 1
                        positive_negative_lst.append(1.0)
                for i in range(len(tmp_ok)):
                    if tmp_ok[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                        is_negative = False
                        ok_count_tmp += 1
                        positive_negative_lst.append(1.0)
                for i in range(len(tmp_junk)):
                    if tmp_junk[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                        is_negative = False
                        junk_count_tmp += 1
                if is_negative:
                    negative_count_tmp += 1
                    positive_negative_lst.append(0)
            tmp_precision = (good_count_tmp+ok_count_tmp)/float(good_count_tmp+ok_count_tmp+negative_count_tmp)
            print 'Good: ', good_count_tmp
            print 'OK: ', ok_count_tmp
            print 'Precision: ', tmp_precision

            tmp_file.close()
            new_output_file.write(str(len(tmp_result)))
            new_output_file.write(',')
            new_output_file.write(str(good_count_tmp+ok_count_tmp))
            new_output_file.write(',')
            new_output_file.write(str(tmp_precision))
            new_output_file.write(',')
            new_output_file.write(str(len(tmp_good)+len(tmp_ok)))
            new_output_file.write(',')
            new_output_file.write(str(Kendall_tau(positive_negative_lst)))
            new_output_file.write('\n')

    output_file.close()
    new_output_file.close()
    #### output a namelist
    namelist_file = open(top_dir + 'namelist.txt', 'w')
    for i in range(len(target_img_dir_list)):
        namelist_file.write(str(i+1) + ': ')
        namelist_file.write(target_img_dir_list[i] + '\n')
    namelist_file.close()
