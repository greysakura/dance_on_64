__author__ = 'LIMU_North'

import numpy as np
import pylab as pl
import os
import logging

top_dir = 'C:/Cassandra/python_oxford/'
evaluation_dir = top_dir + 'evaluation/'
query_goto_dir = 'C:/Cassandra/query_object/'

logging.basicConfig(filename = os.path.join(os.getcwd(), 'mAP_log.txt'),  level = logging.INFO, filemode = 'w',
                    format = '%(asctime)s - %(levelname)s: %(message)s')

logging.info('---------- mAP of test -----------')
logging.info('Method selected: IR-SVM')

######### 2014/06/25

try:
    os.stat(evaluation_dir)
except:
    os.mkdir(evaluation_dir)

def check_list_AP(AP_input):

    if type(AP_input) is list:
        input_mat = np.array(AP_input, np.int32)
    elif type(AP_input) is np.ndarray:
        input_mat = AP_input
        if len(AP_input.shape) >1 :
            input_mat = input_mat.flatten()

    old_precision = 1.0
    old_recall = 0.0
    ap = 0.0
    j = 0
    intersect = 0.0
    precision_list = [1.0]
    recall_list = [0.0]
    for i in range(input_mat.shape[0]):
        if input_mat[i] == 0:
           continue
        if input_mat[i] == 1:
            intersect +=1.0
            recall = intersect / np.where(input_mat == 1)[0].shape[0]
            precision = intersect / (j + 1.0)

            ap += (recall - old_recall) * ((old_precision + precision)/2.0)

            old_recall = recall
            old_precision = precision
            precision_list.append(precision)
            recall_list.append(recall)
        j += 1
    return ap, recall_list, precision_list





target_img_name_list = []
## list inside a list
target_img_matching_img_list = []
# target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
target_img_list = open(query_goto_dir + 'target_img_list.txt', 'r')
for line in target_img_list:
    target_img_name_list.append((line.split('.')[0]).split('/')[-1])
    # print line[:-1]
target_img_list.close()
print len(target_img_name_list)


csv_file = open(top_dir + 'positive_or_not_IR_SVM.csv','r')
# csv_file = open('C:/Cassandra/test_results/140629/test03_SVM/positive_or_not_DQE.csv','r')
# csv_file = open(top_dir + 'positive_or_not_Ranking_SVM.csv','r')
# csv_file = open(top_dir + 'positive_or_not_SV.csv','r')
# csv_file = open(top_dir + 'positive_or_not_RV.csv','r')
# csv_file = open(top_dir + 'positive_or_not_DQE.csv','r')
# csv_file = open(top_dir + 'positive_or_not.csv','r')
line_push = []
for line in csv_file:
    line_push.append(np.int32(line.split(',')))
all_mat = np.array(line_push)
print all_mat.shape
csv_file.close()
mAP_list = []

for i in range(all_mat.shape[1]):
    logging.info('--- Query number %d ---', (i+1))
    logging.info('Query name: ' + str(target_img_name_list[i]))
    print target_img_name_list[i]
    tmp_0_or_1 = all_mat[:,i]
    print tmp_0_or_1[0:20]
    tmp_mAP, tmp_recall, tmp_precision =  check_list_AP(tmp_0_or_1)
    logging.info('tmp precision: ' + str(tmp_precision).strip('[]'))
    logging.info('AP: ' + str(tmp_mAP).strip('[]'))
    print 'P: ', tmp_precision
    print 'AP: ', tmp_mAP
    mAP_list.append(tmp_mAP)
    pl.clf()
    pl.plot(tmp_recall, tmp_precision,'b-o', label='Logistic regression: Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    # pl.title('%s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
    # pl.title('tf-idf weighting, %s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
    # pl.title('tf-idf weighting+SV+linear-SVM, %s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
    pl.legend(loc="upper right", numpoints = 1)
    pl.savefig(evaluation_dir + target_img_name_list[i] + '.jpg')
    # pl.show()
    # pass

# print range(1,11,1)
mAP_mat = np.array(mAP_list)
# print
print 'test mAP: ', mAP_mat.sum()/mAP_mat.shape[0]
test_mAP = mAP_mat.sum()/mAP_mat.shape[0]
logging.info('---------------------------')
logging.info('test mAP: ' + str(test_mAP))
logging.shutdown()
# print
input_list = [1,0,0,1,0,1]
input_list = np.array(input_list)


# precision, recall, area = check_list_AP(input_list, None)
# print area
#
# pl.clf()
# pl.plot(recall, precision, label='Precision-Recall curve')
# pl.xlabel('Recall')
# pl.ylabel('Precision')
# pl.ylim([0.0, 1.05])
# pl.xlim([0.0, 1.0])
# pl.title('Precision-Recall example: AUC=%0.2f' % area)
# pl.legend(loc="lower left")
# pl.show()
# pl.clf()
# pl.plot(recall, precision, label='Precision-Recall curve')
# pl.xlabel('Recall')
# pl.ylabel('Precision')
# pl.ylim([0.0, 1.05])
# pl.xlim([0.0, 1.0])
# pl.title('Precision-Recall example: AUC=%0.2f' % area)
# pl.legend(loc="lower left")
# pl.show()