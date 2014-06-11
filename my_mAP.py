__author__ = 'LIMU_North'

import numpy as np
import pylab as pl
def check_list_AP(AP_input,total_num):

    if type(AP_input) is list:
        input_mat = np.array(AP_input, np.int32)
    elif type(AP_input) is np.ndarray:
        input_mat = AP_input
        if len(AP_input.shape) >1 :
            input_mat = input_mat.flatten()
    input_P_mat = np.array(range(1, input_mat.sum()+1,1), np.float64)


    print (np.where(input_mat == 1)[0])
    if total_num == None:
        tmp_recall = input_P_mat/input_mat.sum()
        # print 'recall: ', tmp_recall
        tmp_precision = tmp_recall / (np.where(input_mat == 1)[0] + 1)
        # print 'precision: ', tmp_precision
        if input_mat.sum() !=0:
            return tmp_precision, tmp_recall, (tmp_recall / (np.where(input_mat == 1)[0] + 1)).sum()
        else:
            return tmp_precision, tmp_recall, 0
    else:
        tmp_recall = input_P_mat/total_num
        # print 'recall: ', tmp_recall
        tmp_precision = tmp_recall / (np.where(input_mat == 1)[0] + 1)
        # print 'precision: ', tmp_precision
        return tmp_precision, tmp_recall, (tmp_recall/(np.where(input_mat == 1)[0] + 1)).sum()







csv_file = open('C:/Cassandra/test_results/140529/sift2000_357_8192_tf_idf/positive_or_not.csv','r')
line_count = 0
line_push = []
for line in csv_file:
    line_count += 1
    line_push.append(np.int32(line.split(',')))
all_mat = np.array(line_push)
print all_mat
print all_mat.shape

mAP_list = []

for i in range(all_mat.shape[1]):
    tmp_0_or_1 = all_mat[:,i]
    tmp_precision, tmp_recall, tmp_mAP =  check_list_AP(tmp_0_or_1, None)
    mAP_list.append(tmp_mAP)
    # pl.plot(tmp_recall, tmp_precision, label='Precision-Recall curve')
    # pl.xlabel('Recall')
    # pl.ylabel('Precision')
    # pl.ylim([0.0, 1.05])
    # pl.xlim([0.0, 1.0])
    # pl.title('Precision-Recall example: AUC=%0.2f' % tmp_mAP)
    # pl.legend(loc="lower left")
    # pl.show()

print range(1,11,1)
mAP_mat = np.array(mAP_list)
print
print mAP_mat.sum()/mAP_mat.shape[0]
print
input_list = [1,0,0,1,0,1]
# print type(input_list)

precision, recall, area = check_list_AP(input_list, None)
print area

# pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()
# pl.clf()
# pl.plot(recall, precision, label='Precision-Recall curve')
# pl.xlabel('Recall')
# pl.ylabel('Precision')
# pl.ylim([0.0, 1.05])
# pl.xlim([0.0, 1.0])
# pl.title('Precision-Recall example: AUC=%0.2f' % area)
# pl.legend(loc="lower left")
# pl.show()