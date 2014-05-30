__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

top_dir = 'C:/Cassandra/Paris_6k/'
dataset_dir = top_dir + 'dataset/paris_all/'
query_goto_dir = top_dir + 'paris_query_object/'
ground_truth_dir = top_dir + 'paris_ground_truth/'

if __name__ == "__main__":
    target_name = ['defense', 'eiffel', 'invalides', 'louvre', 'moulinrouge', 'museedorsay', 'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']

    print target_name
    print len(target_name)
    ## check if the dir exsit. If not, make one.
    if not os.path.exists(query_goto_dir):
                os.makedirs(query_goto_dir)
    ##

    target_img_list = open(query_goto_dir + 'target_img_list.txt', 'w')

    for i in range(len(target_name)):
        query_append = '_query.txt'
        for j in range(5):
            print
            query_file_tmp = open(ground_truth_dir + target_name[i] + '_' + str(j+1) + '_query.txt', 'r')
            line = query_file_tmp.readline()
            query_name_tmp = line.split(' ')[0][5:]
            print line[:-1]
            print 'query_name_tmp: ', query_name_tmp
            query_file_tmp.close()
            ## read the image file from dataset
            print dataset_dir + '/paris' +  query_name_tmp + '.jpg'
            original_img_tmp = cv2.imread(dataset_dir + '/paris' +  query_name_tmp + '.jpg')
            print original_img_tmp.shape



            ## get the query region information
            c0 = float(line.split(' ')[2])
            c1 = float(line.split(' ')[4])
            r0 = float(line.split(' ')[1])
            r1 = float(line.split(' ')[3])

            query_object_img_tmp = original_img_tmp[c0:c1, r0:r1]
            query_object_dir_tmp = query_goto_dir + target_name[i] + '_' + str(j+1) + '_query.jpg'
            ## write into image list
            target_img_list.write(query_goto_dir + target_name[i] + '_' + str(j+1) + '_query.jpg')
            target_img_list.write('\n')

            cv2.imshow(query_object_dir_tmp, query_object_img_tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            ## write down this query object image.

            cv2.imwrite(query_object_dir_tmp, query_object_img_tmp)
    target_img_list.close()
