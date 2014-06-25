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
    this_ap = 0.0
    old_recall = 0.0
    old_precision = 1.0
    intersect_size = 0
    j = 0

    recall_list = [old_recall]
    precision_list = [old_precision]

    if type(AP_input) is list:
        input_mat = np.array(AP_input, np.int32)
    elif type(AP_input) is np.ndarray:
        input_mat = AP_input
        if len(AP_input.shape) >1 :
            input_mat = input_mat.flatten()
    # print np.where(input_mat == 1)[0].shape[0]
    # raw_input('sdf')
    total_positive_num = np.where(input_mat == 1)[0].shape[0]
    print total_positive_num
    print input_mat.shape[0]
    for i in range(input_mat.shape[0]):

        recall_tmp = intersect_size / np.float64(total_positive_num)
        precision_tmp = intersect_size / (j + 1.0)
        recall_list.append(recall_tmp)
        precision_list.append(precision_tmp)

        this_ap += (recall_tmp - old_recall)*((old_precision + precision_tmp)/2.0)
        print this_ap

        old_recall = recall_tmp
        old_precision = precision_tmp
    print this_ap

    raw_input('df')

    return  precision_list, recall_list, this_ap



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



# csv_file = open('C:/Cassandra/test_results/140620/sift160_10k_tf_idf_SV/positive_or_not.csv','r')
csv_file = open(top_dir + 'positive_or_not.csv','r')
line_push = []
for line in csv_file:
    line_push.append(np.int32(line.split(',')))
all_mat = np.array(line_push)
print all_mat.shape

mAP_list = []

for i in range(all_mat.shape[1]):
    # print target_img_name_list[i]
    tmp_0_or_1 = all_mat[:,i]
    print tmp_0_or_1[0:10]
    raw_input('asdfasdf')
    tmp_precision, tmp_recall, tmp_mAP =  check_list_AP(tmp_0_or_1)
    # print 'recall: ', tmp_recall
    # precision_for_draw = [0]
    # precision_for_draw.extend(tmp_precision)
    # recall_for_draw = [0]
    # recall_for_draw.extend(tmp_recall)
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
    pass

# print range(1,11,1)
mAP_mat = np.array(mAP_list)
print mAP_list
raw_input('sdf')
print
print 'test mAP: ', mAP_mat.sum()/mAP_mat.shape[0]
print
input_list = [1,0,0,1,0,1]
# print type(input_list)

precision, recall, area = check_list_AP(input_list)
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