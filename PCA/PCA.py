import numpy as np
import matplotlib.pyplot as plt



class PCA(object):
    def __init__(self):
        self.data = None
        self.lowdata = None
    def KL(self, data, num):
        """
        函数说明：PCA算法实现
        parameters:
            dataMat -用于进行PCA操作的数据集
            topNfeat -应用的N个特征
        return:
            lowDataMat -降维后的数据集
            reconMat -重构的数据集
        """
        self.data = data
        datamat = np.mat(data)
        meanVals = np.mean(datamat, axis = 0)
        data = datamat - meanVals
        Scov = np.zeros((3, 3))
        N = len(datamat)
        for i in range(N):
            cur = datamat[i, :].T * datamat[i, :]
            Scov += cur/N
        # print(Scov)
        eigvals, eigvecs = np.linalg.eig(Scov)
        eigind = np.argsort(-eigvals)
        mainind = eigind[:num]
        maineigvals = eigvals[mainind]
        maineigvecs = eigvecs[:, mainind]
        lowDataMat = np.dot(data, maineigvecs)
        reconMat = np.dot(lowDataMat, maineigvecs.T) + meanVals
        return lowDataMat, reconMat

def loadDataSet(fileName, delim = ','):
    """
    函数说明：加载数据集
    parameters:
       fileName -数据集名称
       delim -分隔符
    return:
       mat(datArr) -数据矩阵
    """   
    with open(fileName, 'r', encoding='utf-8') as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [list(map(float, line[1:])) for line in stringArr]
        label = [0 if line[0]=='男' else 1 for line in stringArr]
    return np.mat(datArr), label

def main():
    filename = 'data.csv'
    data, label = loadDataSet(filename)
    solve = PCA()
    solve.KL(data, 2)
if __name__ == '__main__':
    main()
