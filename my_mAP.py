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

def check_list_AP(AP_input,total_num):

    if type(AP_input) is list:
        input_mat = np.array(AP_input, np.int32)
    elif type(AP_input) is np.ndarray:
        input_mat = AP_input
        if len(AP_input.shape) >1 :
            input_mat = input_mat.flatten()
    input_P_mat = np.array(range(1, input_mat.sum()+1,1), np.float64)
    # print input_P_mat

    if total_num == None:
        tmp_recall = input_P_mat/input_mat.sum()
        # print 'recall: ', tmp_recall
        tmp_precision = input_P_mat / (np.where(input_mat == 1)[0] + 1)
        print
        # print (np.where(input_mat == 1)[0] + 1)
        print 'precisions:', tmp_precision
        # print 'precision: ', tmp_precision
        print 'AP: ', np.average(tmp_precision)
        if input_mat.sum() !=0:
            return tmp_precision, tmp_recall, np.average(tmp_precision)
        else:
            return tmp_precision, tmp_recall, 0
    else:
        tmp_recall = input_P_mat/total_num
        # print 'recall: ', tmp_recall
        tmp_precision = tmp_recall / (np.where(input_mat == 1)[0] + 1)
        # print 'precision: ', tmp_precision

        return tmp_precision, tmp_recall, np.average(tmp_precision)



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



csv_file = open('C:/Cassandra/test_results/140613/sift3000_238_tf_idf_8192/positive_or_not.csv','r')
line_count = 0
line_push = []
for line in csv_file:
    line_count += 1
    line_push.append(np.int32(line.split(',')))
all_mat = np.array(line_push)
print all_mat.shape

mAP_list = []

for i in range(all_mat.shape[1]):
    # print target_img_name_list[i]
    tmp_0_or_1 = all_mat[:,i]
    tmp_precision, tmp_recall, tmp_mAP =  check_list_AP(tmp_0_or_1, None)
    # print 'recall: ', tmp_recall
    precision_for_draw = [0]
    precision_for_draw.extend(tmp_precision)
    recall_for_draw = [0]
    recall_for_draw.extend(tmp_recall)
    mAP_list.append(tmp_mAP)
    pl.clf()
    pl.plot(recall_for_draw, precision_for_draw, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('%s: AUC=%0.2f' % (target_img_name_list[i],tmp_mAP))
    pl.legend(loc="lower left")
    pl.savefig(evaluation_dir + target_img_name_list[i] + '.jpg')
    # pl.show()
    pass

# print range(1,11,1)
mAP_mat = np.array(mAP_list)
print
print 'test mAP: ', mAP_mat.sum()/mAP_mat.shape[0]
print
input_list = [1,0,0,1,0,1]
# print type(input_list)

precision, recall, area = check_list_AP(input_list, None)
print area

pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
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