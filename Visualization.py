__author__ = 'LIMU_North'

import cv2
import numpy as np
import os
import math
import shutil

top_dir = 'C:/Cassandra/python_oxford/'
query_goto_dir = 'C:/Cassandra/query_object/'
TF_IDF_ranking_dir = top_dir + 'TF_IDF_ranking/'
SV_reranking_dir = top_dir + 'SV_reranking/'
DQE_reranking_dir = top_dir + 'DQE_reranking/'
Ranking_SVM_reranking_dir = top_dir + 'Ranking_SVM_reranking/'
dirVisualization = top_dir + 'Visual/'
top_retrieval_num = 5062

try:
    os.stat(dirVisualization)
except:
    os.mkdir(dirVisualization)

target_img_dir_list = []
target_query_list = []
target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
for line in target_img_list:
    target_img_dir_list.append(line[:-1].split('/')[-1])
    target_query_list.append(line.split('\n')[0])
target_img_list.close()


imageLevelFile = open(top_dir + 'imageLevel_DQE.csv', 'r')

listImageLevel = []
for line in imageLevelFile:
    tmpline = line.split('\n')[0]
    listImageLevel.append(tmpline.split(','))
imageLevelFile.close()
matImageLevel_DQE = np.array(listImageLevel, int)


imageLevelFile = open(top_dir + 'imageLevel_Ranking_SVM.csv', 'r')
listImageLevel = []
for line in imageLevelFile:
    tmpline = line.split('\n')[0]
    listImageLevel.append(tmpline.split(','))
imageLevelFile.close()
matImageLevel_Ranking_SVM = np.array(listImageLevel, int)


numTopTaken = 20


# try:
#     os.stat(dirVisualization + 'TF_IDF_ranking/')
# except:
#     os.mkdir(dirVisualization + 'TF_IDF_ranking/')
# for query_i in range(len(target_img_dir_list)):
#     #### create new dir
#     tmp_dir = dirVisualization + 'TF_IDF_ranking/' +target_img_dir_list[query_i].split('.')[0] + '/'
#     print tmp_dir
#     try:
#         os.stat(tmp_dir)
#     except:
#         os.mkdir(tmp_dir)
#     tmp_result_file = open(TF_IDF_ranking_dir + target_img_dir_list[query_i].split('.')[0] + '_TF_IDF_ranking.txt', 'r')
#
#     new_query_dir_str = tmp_dir + 'query_' + str(query_i + 1) + '.jpg'
#     shutil.copy2(target_query_list[query_i], new_query_dir_str)
#     for i in range(numTopTaken):
#         line = tmp_result_file.readline()
#         new_img_dir_str = tmp_dir + 'result_' + str(i+1) + '.jpg'
#         shutil.copy2(line.split('\n')[0], new_img_dir_str)
#     tmp_result_file.close()

#
# try:
#     os.stat(dirVisualization + 'SV_reranking/')
# except:
#     os.mkdir(dirVisualization + 'SV_reranking/')
# for query_i in range(len(target_img_dir_list)):
#     #### create new dir
#     tmp_dir = dirVisualization + 'SV_reranking/' +target_img_dir_list[query_i].split('.')[0] + '/'
#     print tmp_dir
#     try:
#         os.stat(tmp_dir)
#     except:
#         os.mkdir(tmp_dir)
#     tmp_result_file = open(SV_reranking_dir + target_img_dir_list[query_i].split('.')[0] + '_SV_reranking.txt', 'r')
#
#     new_query_dir_str = tmp_dir + 'query_' + str(query_i + 1) + '.jpg'
#     shutil.copy2(target_query_list[query_i], new_query_dir_str)
#     for i in range(numTopTaken):
#         line = tmp_result_file.readline()
#         new_img_dir_str = tmp_dir + 'result_' + str(i+1) + '.jpg'
#         shutil.copy2(line.split('\n')[0], new_img_dir_str)
#     tmp_result_file.close()


needed = False
if needed:
    #### read results DQE
    try:
        os.stat(dirVisualization + 'DQE/')
    except:
        os.mkdir(dirVisualization + 'DQE/')
    for query_i in range(len(target_img_dir_list)):
        #### create new dir
        tmp_dir = dirVisualization + 'DQE/' +target_img_dir_list[query_i].split('.')[0] + '/'
        print tmp_dir
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir)
        tmp_result_file = open(DQE_reranking_dir + target_img_dir_list[query_i].split('.')[0] + '_DQE_reranking.txt', 'r')

        new_query_dir_str = tmp_dir + 'query_' + str(query_i + 1) + '.jpg'
        shutil.copy2(target_query_list[query_i], new_query_dir_str)
        for i in range(numTopTaken):
            line = tmp_result_file.readline()
            new_img_dir_str = tmp_dir + 'result_' + str(i+1) + '.jpg'
            shutil.copy2(line.split('\n')[0], new_img_dir_str)
        tmp_result_file.close()

    #### read results Ranking_SVM
    try:
        os.stat(dirVisualization + 'Ranking_SVM/')
    except:
        os.mkdir(dirVisualization + 'Ranking_SVM/')

    for query_i in range(len(target_img_dir_list)):
        #### create new dir
        tmp_dir = dirVisualization + 'Ranking_SVM/' + target_img_dir_list[query_i].split('.')[0] + '/'
        print tmp_dir
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir)

        tmp_result_file = open(Ranking_SVM_reranking_dir + target_img_dir_list[query_i].split('.')[0] + '_Ranking_SVM.txt', 'r')

        new_query_dir_str = tmp_dir + 'query_' + str(query_i + 1) + '.jpg'
        shutil.copy2(target_query_list[query_i], new_query_dir_str)
        for i in range(numTopTaken):
            line = tmp_result_file.readline()
            new_img_dir_str = tmp_dir + 'result_' + str(i+1) + '.jpg'
            shutil.copy2(line.split('\n')[0], new_img_dir_str)
        tmp_result_file.close()

    SV_result_img_dir = top_dir + 'SV_result_img/'
    try:
        os.stat(SV_result_img_dir)
    except:
        os.mkdir(SV_result_img_dir)

    for query_i in range(len(target_img_dir_list)):
        print 'Query number: ', (query_i+1)

        SV_result_file  = open(top_dir + 'SV_verified/' +  target_img_dir_list[query_i].split('.')[0] + '_SV_result.txt', 'r')
        tmp_SV_result_img_dir = SV_result_img_dir + target_img_dir_list[query_i].split('.')[0] + '/'
        try:
            os.stat(tmp_SV_result_img_dir)
        except:
            os.mkdir(tmp_SV_result_img_dir)

        SV_result_file.readline()
        tmp_num = 0
        for line in SV_result_file:
            new_img_dir_str = tmp_SV_result_img_dir + 'SV_result_' + str(tmp_num+1) + '.jpg'
            tmp_num += 1
            shutil.copy2(line.split('\n')[0], new_img_dir_str)
        SV_result_file.close()

#### read positive or not csv

## method selected
# tmpMethod = 'Ranking_SVM'
# tmpMethod = 'DQE'

###### extra visual

##MethodList = ['DQE', 'Ranking_SVM']

MethodList = ['TF_IDF_ranking', 'SV_reranking','DQE', 'Ranking_SVM']

# MethodList = ['SV_reranking']

for tmpMethod in MethodList:

    positive_or_not_file = open(top_dir + 'positive_or_not_' + tmpMethod + '.csv', 'r')
    lstPositiveNegative = []
    for line in positive_or_not_file:
        lstPositiveNegative.append(line.split('\n')[0].split(','))
    positive_or_not_file.close()

    matPositiveNegative = np.array(lstPositiveNegative, np.int32)

    img_range = 10
    query_ID = 3
    tmp_lst = matPositiveNegative[0:img_range, query_ID].tolist()
    print tmp_lst

    print target_query_list[query_ID]

    wantedHeight = 480
    myThickness = 25

    #### Draw original query
    total_img = cv2.imread(target_query_list[query_ID])
    newHigh = int(float(wantedHeight) * total_img.shape[1]/total_img.shape[0])
    total_img = cv2.resize(total_img, (newHigh,wantedHeight))
    cv2.rectangle(total_img,(0,0), (total_img.shape[1], total_img.shape[0]), (0,255,255),thickness = myThickness)
    # cv2.imshow('asdasdf', total_img)
    # cv2.waitKey(0)
    # raw_input('asdf')


    for i in range(img_range):
        tmp_img = cv2.imread(dirVisualization + tmpMethod + '/' + target_img_dir_list[query_ID].split('.')[0] + '/result_' + str(i+1) + '.jpg')


        newHigh = int(float(wantedHeight) * tmp_img.shape[1]/tmp_img.shape[0])
        tmp_img = cv2.resize(tmp_img, (newHigh,wantedHeight))

        ## paint
        if 1 == tmp_lst[i]:
            cv2.rectangle(tmp_img,(0,0), (tmp_img.shape[1], tmp_img.shape[0]), (0,0,255),thickness = myThickness)
        elif 0 == tmp_lst[i]:
            cv2.rectangle(tmp_img,(0,0), (tmp_img.shape[1], tmp_img.shape[0]), (0,255,0),thickness = myThickness)
        else:
            cv2.rectangle(tmp_img,(0,0), (tmp_img.shape[1], tmp_img.shape[0]), (255,0,0),thickness = myThickness)

        if None == total_img:
            total_img = tmp_img.copy()
        else:
            total_img = np.hstack((total_img, tmp_img))
        # cv2.imshow('tmp',tmp_img)
        # cv2.waitKey(0)

    ## save total
    cv2.imwrite(dirVisualization + tmpMethod + '/' + target_img_dir_list[query_ID].split('.')[0] + '/' + target_img_dir_list[query_ID].split('.')[0] +
                '_total_' + tmpMethod + '.jpg', total_img)