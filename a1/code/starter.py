import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def loadData():
    with np.load('notMNIST.npz') as data :
        # Data (data['images']) is image intensity matrix 0-255
        # Target (data['labels']) is a number from 0 to 9
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9

        # get index array of all correctly labelled data
        dataIndx = (Target==posClass) + (Target==negClass)

        # normalize data intensity 0-1
        Data = Data[dataIndx]/255.

        # Target set to a column vector of only correctly labelled data
        Target = Target[dataIndx].reshape(-1, 1)
        # Classification of target data
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0

        # seed the random number generator for repeatable results
        np.random.seed(421)

        # Shuffle the order of the data and target
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]

        # Set the training data, validation data, and test data
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def parseData(trainData, trainTarget):
    num_data = trainData.shape[0]
    X = trainData.reshape(num_data, -1)
    y = trainTarget
    return X, y

# SET TRAINING, VALIDATION, TEST DATA TO GLOBAL VARS
trainingData, validData, testData, trainingLabels, validTarget, testTarget = loadData()
X_valid, y_valid = parseData(validData, validTarget)
X_test, y_test = parseData(testData, testTarget)

# ------------------------------------------------------------------------------
# LOSS AND ACCURACY FUNCTIONS

def MSE(w, b, X, y, reg):
    N = np.size(X, 0)   # number of rows in X
    L_D = (1/(2*N)) * (np.linalg.norm(X @ w + b - y)**2)
    L_W = (reg/2) * (np.linalg.norm(w)**2)
    return L_D + L_W

def crossEntropyLoss(w, b, X, y, reg):
    N = np.size(X, 0)   # number of datapoints
    z = X @ w + b
    ones = np.ones(N)
    CE = (1/N) * ((1-y).T @ z + ones.T @ np.log(1 + np.exp(-z)))[0,0]
    WD = (reg/2) * (np.linalg.norm(w)**2)
    return CE + WD

def classificationAccuracy(X, y, w, b):
    correct = 0.0
    for n in range(X.shape[0]):
        y_hat = X[n,:] @ w + b
        y_label = y[n,:]

        # Classify the linear regression output
        if y_hat >= 0.5:
            y_hat = 1
        else:
            y_hat = 0

        # Verify classification accuracy
        if y_hat == y_label:
            correct = correct + 1.0

    # Return classification accuracy
    return correct/X.shape[0]

def normalEquation(X, y, reg):
    # add [1 X] for bias term
    X = np.insert(X, 0, 1, axis=1)
    N = np.size(X, 0)   # number of rows in X

    pseudo_inv = np.linalg.inv(X.T @ X + N*reg*np.identity(X.shape[1])) @ X.T
    w_aug = pseudo_inv @ y
    b = w_aug[0,0]              # bias is constant for entire row 0
    w = w_aug[1:,:]
    return w, b

def measurePerformance(w, b, X, y, reg, lossType):
    # Measure Training, Validation, and Test Performance
    if lossType=="MSE":
        train_loss = MSE(w, b, X, y, reg)
        valid_loss = MSE(w, b, X_valid, y_valid, reg)
        test_loss = MSE(w, b, X_test, y_test, reg)
    elif lossType=="CE":
        train_loss = crossEntropyLoss(w, b, X, y, reg)
        valid_loss = crossEntropyLoss(w, b, X_valid, y_valid, reg)
        test_loss = crossEntropyLoss(w, b, X_test, y_test, reg)
    train_accuracy = classificationAccuracy(X, y, w, b)
    valid_accuracy = classificationAccuracy(X_valid, y_valid, w, b)
    test_accuracy = classificationAccuracy(X_test, y_test, w, b)

    return train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy
