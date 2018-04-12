# coding:utf-8

from numpy import *

def predict(W, X):
    return W * X.T

def batchPegasos(dataSet, labels, lam, T, k):
    m, n = shape(dataSet)
    w = zeros(n)
    dataIndex = range(m)
    for t in range(1, T + 1):
        wDelta = mat(zeros(n))
        eta = 1.0 / (lam * t)
        random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, dataSet[i, :])
            if labels[i] * p < 1:
                wDelta += labels[i] * dataSet[i, :].A
        w = (1.0 - 1 / t) * w + (eta / k ) * wDelta
    return w