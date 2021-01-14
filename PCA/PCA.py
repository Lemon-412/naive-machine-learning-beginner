import numpy as np
from copy import deepcopy


class PCA(object):
    def __init__(self, data):
        """
        :param data: 用于进行PCA操作的数据集
        """
        data = deepcopy(data)
        self.__data = np.mat(data)
        self.__m, self.__n = self.__data.shape

    def kl(self, num):
        """
        使用kl进行数据降维
        :param num: 应用的特征个数
        :return: lowDataMat-降维后的数据集, reconMat -重构的数据集
        """
        for i in range(self.__n):
            x1 = self.__data[:, i].min()
            x2 = self.__data[:, i].max()
            for j in range(self.__m):
                self.__data[j, i] = (self.__data[j, i] - x1) / (x2 - x1)
        scov = np.zeros((self.__n, self.__n))
        for i in range(self.__m):
            cur = self.__data[i, :].T * self.__data[i, :]
            scov += cur / self.__m
        eigvals, eigvecs = np.linalg.eig(scov)
        # print(f"eigvecs:\n {eigvecs}")
        # print(f"eigvals:\n {eigvals}")
        eigind = np.argsort(-eigvals)
        mainind = eigind[:num]
        # maineigvals = eigvals[mainind]
        maineigvecs = eigvecs[:, mainind]
        lowDataMat = np.dot(self.__data, maineigvecs)
        reconMat = np.dot(lowDataMat, maineigvecs.T)
        return lowDataMat, reconMat
