__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

kpts_density = 238

Const_Image_Format = [".jpg", ".bmp", ".png"]
class FileFilt:
    fileList = [""]
    counter = 0
    def __init__(self):
        pass
    def FindFile(self,dirr,filtrate = 1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr,s)
            if os.path.isfile(newDir):
                if filtrate:
                        if newDir and(os.path.splitext(newDir)[1] in Const_Image_Format):
                            self.fileList.append(newDir)
                            self.counter += 1
                else:
                    self.fileList.append(newDir)
                    self.counter += 1


def search_dir_and_create_csv(image_dir, desc_dir, kpts_dir, info_dir):
    keypoint_num = 0
    img_gray = cv2.imread(image_dir,1)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.resize(img_gray, (0,0), fx=0.5, fy=0.5)

    ## change!!! use ORB!!!

    # orb = cv2.ORB(nfeatures= 3400)
    num_feature = int(img_gray.shape[0] * img_gray.shape[1] / kpts_density)
    sift = cv2.SIFT(edgeThreshold=0.01, nfeatures = num_feature)
    # kp, des = orb.detectAndCompute(img_gray, None)
    # kp = orb.detect(img_gray)

    kp, des = sift.detectAndCompute(img_gray, None)

    # img = cv2.drawKeypoints(img_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print type(des[0,0])
    # for i in range(len(kp)):
    #     kp[i].octave = (kp[i].octave % 256)
    #     if kp[i].octave > 8:
    #         kp[i].octave = (kp[i].octave - 256)

    keypoint_num = len(kp)
    print image_dir
    print keypoint_num
    #create dirs for kpts and descriptors
    str_des = '_des.csv'
    str_kpts = '_kpts.csv'

    image_dir_des = desc_dir + ((image_dir.split('.'))[0]).split('/')[-1] + str_des
    image_dir_kpts = kpts_dir + ((image_dir.split('.'))[0]).split('/')[-1] + str_kpts
    image_dir_info = info_dir + ((image_dir.split('.'))[0]).split('/')[-1] + '_info.txt'

    #### write descriptors
    that_file = open(image_dir_des, 'w')
    for i in range(des.shape[0]):
        for j in range(des.shape[1]):
            that_file.write(str(des[i, j]))
            if j < (des.shape[1]-1):
                that_file.write(',')
        that_file.write('\n')
    that_file.close()

    #### write key-points
    that_file = open(image_dir_kpts, 'w')
    for i in range(0, len(kp) - 1):
        that_file.write(str(kp[i].pt[0]))
        that_file.write(str(','))
        that_file.write(str(kp[i].pt[1]))
        that_file.write(str(','))
        that_file.write(str(kp[i].size))
        that_file.write(str(','))
        that_file.write(str(kp[i].angle))
        that_file.write(str(','))
        that_file.write(str(kp[i].response))
        that_file.write(str(','))
        that_file.write(str(kp[i].octave))
        that_file.write(str(','))
        that_file.write(str(kp[i].class_id))
        that_file.write('\n')
    that_file.close()

    #### write image info
    that_file = open(image_dir_info, 'w')
    that_file.write(str(keypoint_num))
    that_file.write(' \n')
    that_file.close()

    return keypoint_num

if __name__ == "__main__":
    ## time

    from time import clock
    start=clock()
    total_kpts_num = 0
    top_dir = 'C:/Cassandra/python_oxford/'
    database_image_dir = top_dir + 'database/'
    database_desc_dir = top_dir + 'database_desc/'
    database_kpts_dir = top_dir + 'database_kpts/'
    database_info_dir = top_dir + 'database_info/'
    try:
        os.stat(database_image_dir)
    except:
        os.mkdir(database_image_dir)
    try:
        os.stat(database_desc_dir)
    except:
        os.mkdir(database_desc_dir)
    try:
        os.stat(database_kpts_dir)
    except:
        os.mkdir(database_kpts_dir)
    try:
        os.stat(database_info_dir)
    except:
        os.mkdir(database_info_dir)

    str_image_index_python_append = 'image_index_python.txt'
    str_image_index = top_dir + str_image_index_python_append
    file_image_index = open(str_image_index, 'w')
    image_search_dir = FileFilt()
    image_search_dir.FindFile(dirr=database_image_dir)
    print(image_search_dir.counter)
    for image_dir in image_search_dir.fileList:
        # print "image_dir: ", image_dir_input
        # print type(image_dir_input)
        if len(image_dir) != 0:
            keypoint_num = search_dir_and_create_csv(image_dir, database_desc_dir, database_kpts_dir, database_info_dir)
            total_kpts_num += keypoint_num
            file_image_index.write(image_dir)
            file_image_index.write(',')
            file_image_index.write(str(keypoint_num))
            file_image_index.write('\n')
    file_image_index.close()

    finish=clock()

    print 'time used: ', int((finish-start)/60), ' minutes ', (finish-start)%60, ' seconds.'
    print 'total kpts: ', total_kpts_num