__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
from time import clock

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/python_oxford/'
    query_goto_dir = 'C:/Cassandra/query_object/'
    ground_truth_dir = top_dir + 'ground_truth_file/'

    SV_ranking_result_dir = top_dir + 'SV_reranking'
    top_retrieval_num = 5062

    target_img_dir_list = []
    target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
    for line in target_img_list:
        target_img_dir_list.append(line[:-1])
    target_img_list.close()
    print 'Number of query images: ', len(target_img_dir_list)

    total_image_retrieved = []
    good_count = []
    ok_count = []
    junk_count = []
    negative_count = []
    positive_total = []
    output_file = open(query_goto_dir + 'evaluation.txt', 'w')

    ## zeros and ones


    positive_or_not = np.zeros((len(target_img_dir_list),top_retrieval_num), np.float64)
    ## read result txt for each query
    for query_i in range(len(target_img_dir_list)):
        tmp_good_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('/')[-1])[:-9] + 'good.txt'
        tmp_ok_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('/')[-1])[:-9] + 'ok.txt'
        tmp_junk_dir = ground_truth_dir = top_dir + 'ground_truth_file/' + (target_img_dir_list[query_i].split('/')[-1])[:-9] + 'junk.txt'
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
        ## count the positives
        positive_total.append(len(tmp_good)+ len(tmp_ok))
        tmp_result_file = open(target_img_dir_list[query_i][:-4] + '.txt', 'r')
        for line_i in range(top_retrieval_num):
            line = tmp_result_file.readline()

            tmp_result.append(line.split('/')[:-1])
            # print line.split('/')[:-1]
            is_negative = True
            for i in range(len(tmp_good)):
                if tmp_good[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    positive_or_not[query_i,line_i] = 1.0
                    is_negative = False
                    good_count_tmp += 1
            for i in range(len(tmp_ok)):
                if tmp_ok[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    positive_or_not[query_i,line_i] = 1.0
                    is_negative = False
                    ok_count_tmp += 1
            for i in range(len(tmp_junk)):
                if tmp_junk[i].find((line.split('/')[-1]).split('.')[0]) == 0:
                    positive_or_not[query_i,line_i] = 0.0
                    is_negative = False
                    junk_count_tmp += 1
            if is_negative:
                positive_or_not[query_i,line_i] = -1.0
                negative_count_tmp += 1
        total_image_retrieved.append(len(tmp_result))
        good_count.append(good_count_tmp)
        ok_count.append(ok_count_tmp)
        junk_count.append(junk_count_tmp)
        negative_count.append(negative_count_tmp)
        print 'good: ',good_count_tmp
        print 'ok: ',ok_count_tmp
        print 'junk: ',junk_count_tmp
        print 'negative: ', negative_count_tmp
        tmp_result_file.close()
        output_file.write(str(good_count_tmp))
        output_file.write(' ')
        output_file.write(str(ok_count_tmp))
        output_file.write(' ')
        output_file.write(str(junk_count_tmp))
        output_file.write(' ')
        output_file.write(str(negative_count_tmp))
        output_file.write('\n')
    output_file.close()

    ## mAP, recall
    mAP_all = np.zeros((1,len(total_image_retrieved)), np.float64)
    recall_all = np.zeros((1,len(total_image_retrieved)), np.float64)

    for i in range(len(total_image_retrieved)):
        mAP_all[0,i] = np.float64(good_count[i] + ok_count[i]) / np.float64(negative_count[i] + good_count[i] + ok_count[i])
        recall_all[0,i] = np.float64(good_count[i] + ok_count[i]) / np.float64(positive_total[i])

    mAP_on_query = np.copy(mAP_all)
    mAP_on_query = np.reshape(mAP_on_query, (-1,5))
    mAP_on_query = np.reshape(mAP_on_query.sum(axis = 1), (1,-1))/5

    recall_on_query = np.copy(recall_all)
    recall_on_query = np.reshape(recall_on_query, (-1,5))
    recall_on_query = np.reshape(recall_on_query.sum(axis = 1), (1,-1))/5


    # ## write the final mAP score
    # final_score_file_mAP = open(top_dir + 'final_score_mAP.txt', 'w')
    # for i in range(len(total_image_retrieved)/5):
    #     for j in range(5):
    #         final_score_file_mAP.write(str(mAP_all[0,i*5+j]))
    #         final_score_file_mAP.write(' ')
    #     final_score_file_mAP.write(str(mAP_on_query[0,i]))
    #     final_score_file_mAP.write('\n')
    # final_score_file_mAP.write(str(mAP_on_query.sum(axis=1)[0]/11))
    # final_score_file_mAP.write('\n')
    # final_score_file_mAP.close()
    #
    # ## write the final recall score
    # final_score_file_recall = open(top_dir + 'final_score_Recall.txt', 'w')
    # for i in range(len(total_image_retrieved)/5):
    #     for j in range(5):
    #         final_score_file_recall.write(str(recall_all[0, i*5+j]))
    #         final_score_file_recall.write(' ')
    #     final_score_file_recall.write(str(recall_on_query[0,i]))
    #     final_score_file_recall.write('\n')
    # final_score_file_recall.write(str(recall_on_query.sum(axis=1)[0]/11))
    # final_score_file_recall.write('\n')
    # final_score_file_recall.close()

    top_some = np.reshape(positive_or_not.sum(axis=1)/top_retrieval_num,(-1,5))
    top_mAP = np.reshape((top_some.sum(axis=1)/5),(-1,1))


    top_some_bigger = np.concatenate((top_some,top_mAP),axis=1)

    positive_or_not_csv =  np.transpose(np.int32(positive_or_not))

    mAP_csv = np.zeros((top_retrieval_num, positive_or_not_csv.shape[1]), np.float64)

    for i in range(top_retrieval_num):
        # print i
        mAP_csv[i,:] = np.float64(positive_or_not_csv[0:(i+1),:].sum(axis = 0)) / np.float64(i+1)
        # print positive_or_not_csv[0:(i+1),:].sum(axis = 0)
    mAP_AP = mAP_csv.sum(axis = 1)/ np.float64(positive_or_not_csv.shape[1])
    mAP_AP = np.reshape(mAP_AP, (-1,1))
    # print mAP_AP.shape
    # print mAP_csv.shape
    # mAP_csv = np.concatenate((mAP_csv,mAP_AP), axis = 1)

    # mAP_csv_file = open(top_dir + 'mAP_series.csv', 'w')
    # ## write title
    # for i in range(mAP_csv.shape[1]-1):
    #     mAP_csv_file.write('Query')
    #     mAP_csv_file.write(str(int(i+1)))
    #     mAP_csv_file.write(',')
    # mAP_csv_file.write('ALL')
    # mAP_csv_file.write('\n')
    #
    # ##write data
    # for i in range(mAP_csv.shape[0]):
    #     for j in range(mAP_csv.shape[1]):
    #         mAP_csv_file.write(str(mAP_csv[i,j]))
    #         if j < (mAP_csv.shape[1] - 1):
    #             mAP_csv_file.write(',')
    #     mAP_csv_file.write('\n')
    #
    # mAP_csv_file.close()

    positive_or_not_file = open(top_dir + 'positive_or_not.csv', 'w')
    for i in range(positive_or_not_csv.shape[0]):
        for j in range(positive_or_not_csv.shape[1]):
            positive_or_not_file.write(str(positive_or_not_csv[i,j]))
            if j < (positive_or_not_csv.shape[1] - 1):
                positive_or_not_file.write(',')
        positive_or_not_file.write('\n')
    positive_or_not_file.close()