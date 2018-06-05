# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet(fileName, delime='\t'):
    stringArr = [line.strip().split(delime) for line in open(fileName).readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


if __name__ == "__main__":
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print shape(lowDMat), reconMat