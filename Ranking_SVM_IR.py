__author__ = 'LIMU_North'
## This is a test for using IR-SVM

import cv2
import numpy as np
import os
import math
from sklearn import svm, linear_model
from time import clock

def IR_SVM(example,label,eta,my_lambda,max_iter):
    ## check if example and label are np.ndarray
    if type(example) != np.ndarray:
        print "Error: input example need to be a np.ndarray"
        exit(1)

    if type(label) != np.ndarray:
        print "Error: input label need to be a np.ndarray"
        exit(1)

    label = label.flatten()
    ## examples are ranked already, we suppose.
    ## construct set S'
    S_02 = []
    BIG_Z = []
    tau = []

    for i in range(example.shape[0]):
        for j in range(i+1,example.shape[0]):
            if label[i] == label[j]:
                continue
            else:
                diff_vector = example[i,:] - example[j,:]
                tau.append(abs(label[i]-label[j]))
                S_02.append(diff_vector)
                if label[i] > label[j]:
                    BIG_Z.append(+1)
                else:
                    BIG_Z.append(-1)
    print "number of inversion pair: ", len(BIG_Z)

    w = np.zeros((example.shape[1]),np.float32)
    while_iter = 0
    while True:
        weight_tmp = np.zeros((example.shape[1]),np.float32)
        for i  in range(len(BIG_Z)):
            judge = BIG_Z[i]* np.dot(w,S_02[i])
            if judge < 1:
                weight_tmp = weight_tmp + tau[i]* BIG_Z[i] * S_02[i].flatten()
        weight_tmp += -2*my_lambda*w
        w += eta * weight_tmp
        while_iter += 1
        if while_iter >= max_iter or np.dot(eta * weight_tmp, eta * weight_tmp)<0.01:
            print "num of iteration: ", while_iter
            break

    ## w is the final output trained weight vector we need.
    return w

if __name__ == "__main__":
    # example = np.random.rand(500,500)
    # example = 1
    # label = np.random.rand(500)
    eta = 0.5
    my_lambda = 0.25
    max_iter = 500
    train_example = np.array(([1,1,0,0.2,0], [0,0,1,0.1,1], [0,1,0,0.4,0], [0,0,1,0.3,0]), np.float32)
    train_label = np.array([3,2,1,1], np.float32)

    test_example = np.array(([1, 0 , 0, 0.2, 1], [1,1,0,0.3,0], [0,0,0,0.2,1], [0,0,1,0.2,0]))

    w = IR_SVM(train_example, train_label, eta, my_lambda, max_iter)
    print w
    print test_example.dot(w)
