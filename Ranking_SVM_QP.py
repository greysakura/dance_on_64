__author__ = 'LIMU_North'

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy

def calcKernelVectorValue(svm, k):
    return np.dot(svm.train_x, svm.train_x[k,:].T)

def calcKernelPointValue(svm, i, j):
    return np.dot(svm.train_x[i,:], svm.train_x[j,:].T)

# calulate kernel value
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in xrange(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = np.exp(diff * diff.T / (-2.0 * sigma**2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
    print 'Starting kernel matrix calc.'
    startTime = time.time()
    numSamples = train_x.shape[0]
    # kernelMatrix = np.mat(np.zeros((numSamples, numSamples)))
    kernelMatrix = np.dot(train_x,train_x.T)
    # for i in xrange(numSamples):
    #     kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    endTime = time.time()
    print 'kernel matrix calc completed. Time used: ', endTime - startTime, ' seconds...'
    return kernelMatrix


# define a struct just for storing variables and data
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet # each row stands for a sample
        self.train_y = labels  # corresponding label

        ## we need to change this...
        self.C = C             # slack variable series
        #########################################

        self.toler = toler     # termination condition for iteration
        self.numSamples = dataSet.shape[0] # number of samples
        self.alphas = np.mat(np.zeros((self.numSamples, 1))) # Lagrange factors for all samples
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


# calculate the error for alpha k
def calcError(svm, alpha_k):
    ## .T: transpose
    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
    # output_k = float(np.multiply(svm.alphas, svm.train_y).T * calcKernelVectorValue(svm, alpha_k) + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


# select alpha j which has the biggest step
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)
    candidateAlphaList = np.nonzero(svm.errorCache[:, 0].A)[0] # mat.A return array
    maxStep = 0; alpha_j = 0; error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    # if came in this loop first time, we select alpha j randomly
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.numSamples))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j


# the inner loop for optimizing alpha i and alpha j
def innerLoop(svm, alpha_i):
    timeStartLoop = time.time()
    # print 'innerloop: ', alpha_i
    error_i = calcError(svm, alpha_i)

    # print 'Error: ', error_i
    ### check and pick up the alpha who violates the KKT condition
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized

    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C[alpha_i]) or\
        (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # step 1: select alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)

        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C[alpha_j], svm.C[alpha_j] + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C[alpha_i])
            H = min(svm.C[alpha_j], svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            timeEndLoop = time.time()
            # print 'Loop time: ', timeEndLoop - timeStartLoop, ' seconds...'
            return 0

        # step 3: calculate eta (the similarity of sample i and j)
        ##################################
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
                  - svm.kernelMat[alpha_j, alpha_j]
        ##################################
        # eta = 2.0 * calcKernelPointValue(svm, alpha_i,alpha_j) - calcKernelPointValue(svm, alpha_i, alpha_i)\
        #       - calcKernelPointValue(svm, alpha_j, alpha_j)
        ##################################
        if eta >= 0:
            timeEndLoop = time.time()
            # print 'Loop time: ', timeEndLoop - timeStartLoop, ' seconds...'
            return 0

        # step 4: update alpha j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # step 5: clip alpha j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            timeEndLoop = time.time()
            # print 'Loop time: ', timeEndLoop - timeStartLoop, ' seconds...'
            return 0

        # step 7: update alpha i after optimizing aipha j
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                                * (alpha_j_old - svm.alphas[alpha_j])

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernelMat[alpha_i, alpha_i] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernelMat[alpha_i, alpha_j]
        ############################################
        # b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
        #                                             * calcKernelPointValue(svm, alpha_i, alpha_i) \
        #                      - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
        #                                             * calcKernelPointValue(svm, alpha_i, alpha_j)
        ############################################
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernelMat[alpha_i, alpha_j] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernelMat[alpha_j, alpha_j]
        ############################################
        # b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
        #                                             * calcKernelPointValue(svm, alpha_i, alpha_j) \
        #                      - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
        #                                             * calcKernelPointValue(svm, alpha_j, alpha_j)
        ############################################
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C[alpha_i]):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C[alpha_j]):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        timeEndLoop = time.time()
        # print 'Loop time: ', timeEndLoop - timeStartLoop, ' seconds...'
        return 1
    else:
        timeEndLoop = time.time()
        # print 'Loop time: ', timeEndLoop - timeStartLoop, ' seconds...'
        return 0


# the main training procedure
def trainSVM(train_x, train_y, train_C, toler, maxIter, kernelOption = ('linear', 1.0)):
    print 'Start training...'
    # calculate training time
    startTime = time.time()

    # init data struct for svm
    svm = SVMStruct(np.mat(train_x), np.mat(train_y), np.mat(train_C), toler, kernelOption)

    print 'SVM construction finished...'
    # start training
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # Iteration termination condition:
    #   Condition 1: reach max iteration
    #   Condition 2: no alpha changed after going through all samples,
    #                in other words, all alpha (samples) fit KKT condition
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        print 'iterCount: ', iterCount
        iterStart = time.time()
        alphaPairsChanged = 0

        # update alphas over all training examples
        if entireSet:
            ## for every i ??
            ## xrange: similar to range
            ## innerloop on i
            for i in xrange(svm.numSamples):
                ## innerLoop is the real loop part?
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1
        # update alphas over examples where alpha is not 0 & not C (not on boundary)
        else:
            nonBoundAlphasList = np.nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C.A))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
            iterCount += 1

        # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        iterEnd = time.time()
        print 'Iteration ', iterCount, ', time used: ', iterEnd - iterStart, ' seconds...'

    print 'Training complete! Took %fs!' % (time.time() - startTime)
    print 'Number of iterations: %d' %iterCount
    return svm


# testing your trained svm model given test set
def testSVM(svm, test_x):
    # calculate training time
    startTime = time.time()
    ####
    prediction = []
    ####
    test_x = np.mat(test_x)
    numTestSamples = test_x.shape[0]

    ## support vectors?
    supportVectorsIndex = np.nonzero(svm.alphas.A > 0)[0]
    supportVectors      = svm.train_x[supportVectorsIndex]
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]

    for i in xrange(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
        ## Here is how they do the prediction.

        predict = kernelValue.T * np.multiply(supportVectorLabels, supportVectorAlphas)
        # predict = kernelValue.T * np.multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        prediction.append((predict.A.flatten())[0])
    print 'Prediction complete! Took %fs!' % (time.time() - startTime)
    return prediction


# # show your trained svm model only available with 2-D data
# def showSVM(svm):
#     if svm.train_x.shape[1] != 2:
#         print "Sorry! I can not draw because the dimension of your data is not 2!"
#         return 1
#
#     # draw all samples
#     for i in xrange(svm.numSamples):
#         if svm.train_y[i] == -1:
#             plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
#         elif svm.train_y[i] == 1:
#             plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')
#
#     # mark support vectors
#     supportVectorsIndex = np.nonzero(svm.alphas.A > 0)[0]
#     for i in supportVectorsIndex:
#         plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')
#
#     # draw the classify line
#     w = np.zeros((2, 1))
#     for i in supportVectorsIndex:
#         w += np.multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
#     min_x = min(svm.train_x[:, 0])[0, 0]
#     max_x = max(svm.train_x[:, 0])[0, 0]
#     y_min_x = float(-svm.b - w[0] * min_x) / w[1]
#     y_max_x = float(-svm.b - w[0] * max_x) / w[1]
#     plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
#     plt.show()

# def train_data_preparation(data_x, data_y, C):
#     print 'Preparing training data for SVM...'
#     train_x = []
#     train_y = []
#     train_C = []
#     for i in range(data_x.shape[0]):
#         for j in range(i+1,data_x.shape[0]):
#             if data_y[i] == data_y[j]:
#                 continue
#             else:
#                 train_C.append(abs(data_y[i]-data_y[j]))
#                 train_C.append(abs(data_y[j]-data_y[i]))
#                 train_x.append(data_x[i,:] - data_x[j,:])
#                 train_x.append(data_x[j,:] - data_x[i,:])
#
#                 if data_y[i] > data_y[j]:
#                     train_y.append(+1)
#                     train_y.append(-1)
#                 else:
#                     train_y.append(-1)
#                     train_y.append(+1)
#     train_x = np.mat(train_x)
#     train_y = np.mat(train_y).T
#     train_C = np.array(train_C)
#     train_C = train_C/train_C.sum()
#     train_C = C * np.mat(train_C).T
#     print 'Number of training examples into SVM: ', train_x.shape[0]
#     return train_x, train_y, train_C

def train_data_preparation(data_x, data_y, C, low_cut = 0.1):
    print 'Preparing training data for SVM...'
    train_x = []
    train_y = []
    train_C = []
    tmp_index = []
    diff_max = 0
    for i in range(data_x.shape[0]):
        for j in range(i+1,data_x.shape[0]):
            if data_y[i] == data_y[j]:
                continue
            else:
               tmp_index.append(((data_y[i] - data_y[j]), i,j))
               if diff_max < abs(data_y[i]-data_y[j]):
                    diff_max = abs(data_y[i] - data_y[j])

    ##  delete element under diff_max/10
    new_index = filter(lambda a: a[0] > (diff_max * low_cut), tmp_index)

    for i in range(len(new_index)):
        train_x.append(data_x[new_index[i][1],:]-data_x[new_index[i][2],:])
        train_x.append(data_x[new_index[i][2],:]-data_x[new_index[i][1],:])
        if (data_y[new_index[i][1]]-data_y[new_index[i][2]])>0:
               train_y.append(1)
               train_y.append(-1)
               train_C.append(data_y[new_index[i][1]]-data_y[new_index[i][2]])
               train_C.append(data_y[new_index[i][1]]-data_y[new_index[i][2]])
        else:
               train_y.append(-1)
               train_y.append(1)
               train_C.append(-(data_y[new_index[i][1]]-data_y[new_index[i][2]]))
               train_C.append(-(data_y[new_index[i][1]]-data_y[new_index[i][2]]))


    train_x = np.mat(train_x)
    train_y = np.mat(train_y).T
    train_C = np.array(train_C)
    train_C = train_C/train_C.sum()
    train_C = C * np.mat(train_C).T
    # print 'train_C: ', train_C
    # print 'train_y: ', train_y
    # print 'train_x: ', train_x
    # raw_input('asdfasdfasdfsadfasdf')
    print 'Number of training examples into SVM: ', train_x.shape[0]
    return train_x, train_y, train_C


if __name__ == "__main__":
    ################## test svm #####################
    ## step 1: load data
    print "step 1: load data..."
    dataSet = []
    labels = []
    fileIn = open('C:/Cassandra/ranking_svm/testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labels.append(float(lineArr[2]))
    fileIn.close()
    #####################
    dataSet = np.mat(dataSet)
    labels = np.mat(labels).T
    ###########################

    train_x = dataSet[0:81, :]
    train_y = labels[0:81, :]
    test_x = dataSet[80:101, :]
    test_y = labels[80:101, :]
    ##################
    data_x = np.array(([1,1,0,0.2,0], [0,0,1,0.1,1], [0,1,0,0.4,0], [0,0,1,0.3,0]), np.float32)
    data_y = np.array([3,2,1,1], np.float32)
    train_x = []
    train_y = []
    train_C = []


    #
    ## step 2: training...
    print "step 2: training..."
    C = 0.6
    toler = 0.001
    maxIter = 50
    # train_x = np.mat(train_x)
    # train_y = np.mat(train_y).T
    # train_C = C * np.mat(train_C).T
    train_x, train_y, train_C = train_data_preparation(data_x, data_y, C)

    svmClassifier = trainSVM(train_x, train_y, train_C, toler, maxIter, kernelOption = ('linear', 0))

    # print train_C
    print train_y.shape
    print svmClassifier.b
    print svmClassifier.alphas.A
    print np.nonzero(svmClassifier.alphas.A.flatten())[0]
    prediction = testSVM(svmClassifier, data_x)
    print prediction

    I = np.array([0,3,1,0])
    J = np.array([0,3,1,2])
    V = np.array([4,5,7,9])
    # A = sparse.coo_matrix((V,(I,J)),shape=(4,4))
    # print A




    # svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))
    #
    # ## step 3: testing
    # print "step 3: testing..."
    # # accuracy = testSVM(svmClassifier, test_x, test_y)
    # prediction = testSVM(svmClassifier, test_x, test_y)
    # print prediction
    # # ## step 4: show the result
    # # print "step 4: show the result..."
    # # print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    # # showSVM(svmClassifier)

    print train_C.shape[0]