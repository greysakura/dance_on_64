__author__ = 'LIMU_North'

import numpy as np
import pylab as pl
import os
import cv2
import time
import psutil


from time import gmtime, strftime

from sklearn.cluster import MiniBatchKMeans, KMeans

if __name__ == '__main__':

    top_dir = 'C:/Cassandra/python_oxford/'
    database_dir = 'C:/Cassandra/oxbuild_images/'
    evaluation_dir = top_dir + 'evaluation/'
    query_goto_dir = 'C:/Cassandra/query_object/'



    csv_file = open('C:/Cassandra/test_results/140629/test01_RBF_C_1_gamma_1/positive_or_not_DQE.csv','r')
    # csv_file = open(top_dir + 'positive_or_not_DQE.csv','r')
    # csv_file = open(top_dir + 'positive_or_not.csv','r')
    line_push = []
    for line in csv_file:
        line_push.append(np.int32(line.split(',')))
    DQE_mat = np.array(line_push)
    print DQE_mat.shape
    line_push = None
    csv_file.close()

    csv_file = open('C:/Cassandra/test_results/140629/test01_RBF_C_1_gamma_1/positive_or_not.csv','r')

    line_push = []
    for line in csv_file:
        line_push.append(np.int32(line.split(',')))
    TF_IDF_mat = np.array(line_push)
    print TF_IDF_mat.shape
    line_push = None
    csv_file.close()

    print TF_IDF_mat[0:9, 9]
    print DQE_mat[0:9,9]

    TF_IDF_ranking_file = open('C:/Cassandra/test_results/140629/test01_RBF_C_1_gamma_1/TF_IDF_ranking/ashmolean_5_query_TF_IDF_ranking.txt','r')
    DQE_ranking_file = open('C:/Cassandra/test_results/140629/test01_RBF_C_1_gamma_1/DQE_reranking/ashmolean_5_query_DQE_reranking.txt','r')



    output_dir = 'C:/Cassandra/enshu2_output/'
    TF_IDF_output = output_dir + 'TF_IDF_output/'
    DQE_output = output_dir + 'DQE_output/'

    try:
        os.stat(output_dir)
    except:
        os.mkdir(output_dir)
    try:
        os.stat(TF_IDF_output)
    except:
        os.mkdir(TF_IDF_output)
    try:
        os.stat(DQE_output)
    except:
        os.mkdir(DQE_output)

    # ## TF_IDF series
    # top_taken = 10
    #
    # for i in range(top_taken):
    #     line = TF_IDF_ranking_file.readline()
    #
    #     img_TF_IDF_tmp = cv2.imread(str(line.split('\n')[0]),1)
    #     if TF_IDF_mat[i,9] == 1:
    #         cv2.rectangle(img_TF_IDF_tmp,(0,0),(img_TF_IDF_tmp.shape[1],img_TF_IDF_tmp.shape[0]),(255,0,0),thickness=20)
    #     else:
    #         cv2.rectangle(img_TF_IDF_tmp,(0,0),(img_TF_IDF_tmp.shape[1],img_TF_IDF_tmp.shape[0]),(0,0,255),thickness=20)
    #     TF_IDF_new_name = 'TF_IDF_ranking_' + str(i) + '.jpg'
    #     cv2.imwrite(TF_IDF_output + TF_IDF_new_name, img_TF_IDF_tmp)
    #
    #     line = DQE_ranking_file.readline()
    #
    #     img_DQE_tmp = cv2.imread(str(line.split('\n')[0]),1)
    #     if DQE_mat[i,9] == 1:
    #         cv2.rectangle(img_DQE_tmp,(0,0),(img_DQE_tmp.shape[1],img_DQE_tmp.shape[0]),(255,0,0),thickness=50)
    #     else:
    #         cv2.rectangle(img_DQE_tmp,(0,0),(img_DQE_tmp.shape[1],img_DQE_tmp.shape[0]),(0,0,255),thickness=50)
    #     DQE_new_name = 'DQE_reranking_' + str(i) + '.jpg'
    #     cv2.imwrite(DQE_output + DQE_new_name, img_DQE_tmp)
    #
    #
    # TF_IDF_ranking_file.close()
    # DQE_ranking_file.close()

    print 'Program start at: ',strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # data_file = open('C:/Cassandra/output_csv_file.csv', 'r')
    #
    # data_list = []
    #
    # for line in data_file:
    #     data_list.append(np.float32(line.split(',')))
    # data_file.close()
    # data_mat = np.array(data_list)
    #
    # np.save('C:/Cassandra/output_csv_file.npy', data_mat)
    #
    # raw_input('stop')
    data_mat = np.load('C:/Cassandra/output_csv_file.npy')

    # print psutil.cpu_percent(interval=1, percpu=True)
    # raw_input('asdfasdf')
    # print psutil.virtual_memory().percent
    # print psutil.cpu_percent(interval=0)
    # raw_input('asdfasdf')


    ## MiniBatchKMeans
    # rng = np.random.RandomState(233)
    # raw_input('stop')
    n_clusters = 10000
    batch_size = 3*n_clusters
    n_init = 20
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,compute_labels=False,
                          n_init=n_init,init_size= 3*batch_size, max_no_improvement=50, verbose=0)

    # kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, max_iter=10,n_jobs = 3)

    print 'KMeans start at: ',strftime("%Y-%m-%d %H:%M:%S", gmtime())
    t0 = time.time()
    # mbk.fit(data_mat)
    # kmeans.fit(data_mat)
    try:
        mbk.fit(data_mat)
    except MemoryError:
        print 'error. present time: ', strftime("%Y-%m-%d %H:%M:%S", gmtime())


    t_mini_batch = time.time() - t0
    print 'minibatchkmeans time used: ',t_mini_batch, 'seconds...'
    # mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    # mbk_means_labels_unique = np.unique(mbk_means_labels)
    print
    raw_input('waiting here...')

    print '...starting output...'

    np.save(top_dir + 'mbk_cluster_centers.npy',mbk.cluster_centers_)

    output_csv_file = open(top_dir + 'mbk_cluster_centers.csv', 'w')
    for i in range(mbk.cluster_centers_.shape[0]):
        for j in range(mbk.cluster_centers_.shape[1]):
            output_csv_file.write(str(mbk.cluster_centers_[i,j]))
            if j < (mbk.cluster_centers_.shape[1]-1):
                output_csv_file.write(',')
        output_csv_file.write('\n')
    output_csv_file.close()

    print mbk_means_cluster_centers.shape

    print 'Program end at: ',strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # print
    #
    # k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, n_jobs = -1)
    # t1 = time.time()
    # k_means.fit(data_mat)
    # t_kmeans = time.time() - t1
    #
    # print t_kmeans
    # print 'mbk: ', mbk.inertia_
    # print 'k-means: ', k_means.inertia_



