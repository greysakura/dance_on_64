__author__ = 'LIMU_North'

from oxford_AP import check_list_AP
import numpy as np
import pylab as pl
import os

top_dir = 'C:/Cassandra/python_oxford/'
evaluation_dir = top_dir + 'evaluation/'
query_goto_dir = 'C:/Cassandra/query_object/'


target_img_name_list = []
## list inside a list
target_img_matching_img_list = []
# target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
for line in target_img_list:
    target_img_name_list.append((line.split('.')[0]).split('/')[-1])
    # print line[:-1]
target_img_list.close()
# print len(target_img_name_list)

##  TF_IDF, SV, DQE, DQE-extra, Ranking_SVM
tmpMethod = ['TF_IDF_ranking', 'SV_reranking', 'DQE', 'Ranking_SVM']

P_or_N_mat_all = []
for i in range(len(tmpMethod)):
    csv_file = open(top_dir + 'positive_or_not_' + tmpMethod[i] + '.csv','r')
    line_push = []
    for line in csv_file:
        line_push.append(np.int32(line.split(',')))
    P_or_N_mat_all.append(np.array(line_push))
    # print P_or_N_mat_all[-1].shape
    csv_file.close()

tmp_query = [3,12,14,44,54]
name_query = ['"all souls 4"','"balliol 3"','"balliol 5"','"magdalen 5"', '"radcliffe camera 5"']
nameMethod = ['first ranking', 'SV re-ranking', 'DQE', 'proposed']
tmp_color = ['b-o', 'r-o', 'g-o', 'y-o']

for query_i in range(len(tmp_query)):
    print target_img_name_list[tmp_query[query_i]]
    pl.clf()
    pl.title('Precision-Recall curve for tested methods on query '+ name_query[query_i])
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.05])
    for method_i in range(len(tmpMethod)):
        # logging.info('--- Query number %d ---', (i+1))
        # logging.info('Query name: ' + str(target_img_name_list[i]))
        print 'tmp method: ', tmpMethod[method_i]

        tmp_0_or_1 = P_or_N_mat_all[method_i][:,tmp_query[query_i]]
        # print tmp_0_or_1[0:20]
        tmp_mAP, tmp_recall, tmp_precision =  check_list_AP(tmp_0_or_1)
        # logging.info('tmp precision: ' + str(tmp_precision).strip('[]'))
        # logging.info('AP: ' + str(tmp_mAP).strip('[]'))
        print 'P: ', tmp_precision
        print 'AP: ', tmp_mAP


        pl.plot(tmp_recall, tmp_precision,tmp_color[method_i], label=nameMethod[method_i])

        # pl.title('%s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
        # pl.title('tf-idf weighting, %s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
        # pl.title('tf-idf weighting+SV+linear-SVM, %s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))

    pl.legend(loc="upper right", numpoints = 1)
    pl.savefig(evaluation_dir + target_img_name_list[tmp_query[query_i]] +'_all_method.jpg')