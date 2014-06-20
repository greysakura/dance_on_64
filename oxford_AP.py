__author__ = 'LIMU_North'

import numpy as np
import pylab as pl
import os

top_dir = 'C:/Cassandra/python_oxford/'
evaluation_dir = top_dir + 'evaluation/'
query_goto_dir = 'C:/Cassandra/query_object/'

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



csv_file = open('C:/Cassandra/test_results/140617/positive_or_not.csv','r')
line_push = []
for line in csv_file:
    line_push.append(np.int32(line.split(',')))
all_mat = np.array(line_push)
print all_mat.shape

mAP_list = []

for i in range(all_mat.shape[1]):
    print target_img_name_list[i]
    tmp_0_or_1 = all_mat[:,i]
    tmp_mAP, tmp_recall, tmp_precision =  check_list_AP(tmp_0_or_1)
    print 'recall: ', tmp_recall
    mAP_list.append(tmp_mAP)
    pl.clf()
    pl.plot(tmp_recall, tmp_precision, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('%s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
    pl.legend(loc="lower left")
    pl.savefig(evaluation_dir + target_img_name_list[i] + '.jpg')
    # pl.show()
    # pass

# print range(1,11,1)
mAP_mat = np.array(mAP_list)
# print
print 'test mAP: ', mAP_mat.sum()/mAP_mat.shape[0]
# print
input_list = [1,0,0,1,0,1]
input_list = np.array(input_list)
print np.where(input_list == 1)[0].shape[0]
#
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