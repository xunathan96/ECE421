# Implementation of a neural network using only Numpy
#   - trained using gradient descent with momentum
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def parseData(data):
    num_data = data.shape[0]
    X = data.reshape(num_data, -1)
    return X

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

# PARSE THE DATA --------------------------------------------------------------------
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
X_train, X_valid, X_test = parseData(trainData), parseData(validData), parseData(testData)
Y_train, Y_valid, Y_test = convertOneHot(trainTarget, validTarget, testTarget)

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(S):
    X = np.copy(S)
    X[S<0] = 0
    return X

def derivative_relu(S):
    dS = np.zeros_like(S)
    dS[S>0] = 1
    return dS

def softmax(S):
    X = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)
    return X

def computeLayer(X, W, b):
    S = X @ W + b
    return S

def avgCE(target, prediction):
    N = prediction.shape[0]
    L = - (1/N) * np.sum(target * np.log(prediction))
    return L

def gradCE(target, prediction):
    N = prediction.shape[0]
    dE_dX = - (1/N) * target / prediction
    return dE_dX

def accuracy(Y, Y_pred):
    j = np.argmax(Y_pred, axis=1)
    i = np.arange(Y.shape[0])
    return np.mean(Y[i, j])
