# coding: utf-8

from numpy import *

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

    def calcEk(self, k):
        fXk = float(multiply(self.alphas, self.labelMat).T * (self.X * self.X[k, :].T)) + self.b
        Ek = fXk - float(self.labelMat[k])
        return Ek

    # 内循环中的启发式方法
    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = nonzero(self.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)

                # 选择具有最大步长的j
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def innerL(self, i):
        Ei = self.calcEk(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[i] - self.alphas[j])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                print "L == H"
                return 0

            eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T
            if eta >= 0:
                print "eta >= 0"
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = clipAlpha(self.alphas[j], H, L)
            # 更新误差缓存
            self.updateEk(j)
            if (abs(self.alphas[j] - alphaJold) < 0.00001):
                print "j not moving enough"
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (alphaJold - self.alphas[j])
            self.updateEk(i)

            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[i, :].T - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[i, :] * self.X[j, :].T
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[j, :].T - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[j, :] * self.X[j, :].T
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            return 1
        else:
            return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    opt = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(opt.m):
                alphaPairsChanged += opt.innerL(i)
            print 'fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = nonzero((opt.alphas.A > 0) * (opt.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += opt.innerL(i)
                print 'non-bound, iter:%d, i:%d pairs changed %d' % (iter, i, alphaPairsChanged)
            iter += 1

        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
            print 'iteration number: %d' % iter
    return opt.b, opt.alphas

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))

    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L

    return aj

#
# 数据集，类别标签，常数C，容错率，退出前最大的循环次数
#
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L == H"
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 设置常数项
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b, alphas

if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('testset.txt')
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    for i in range(100):
        if alphas[i] > 0.0:
            print dataArr[i], labelArr[i]