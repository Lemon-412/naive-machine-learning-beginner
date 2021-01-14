import numpy as np
from copy import deepcopy


class PCA(object):
    def __init__(self, raw_data):
        """
        :param raw_data: 用于进行PCA操作的数据集
        """
        self.__data = np.mat(raw_data)
        self.__data_min_max = None
        self.__normalized_data = None
        self.__pca_restored_data = None
        self.__pca_normalized_data = None
        self.__pca_data_mat = None
        self.__m, self.__n = self.__data.shape


    @property
    def restored_data(self):
        return self.__pca_restored_data

    def pca(self, features):
        """
        使用kl进行数据降维
        :param features: 应用的特征个数
        """
        self.__normalized_data = deepcopy(self.__data)
        self.__data_min_max = np.zeros((self.__n, 2))
        for i in range(self.__n):  # 将各特征进行归一化
            self.__data_min_max[i][0] = self.__normalized_data[:, i].min()
            self.__data_min_max[i][1] = self.__normalized_data[:, i].max()
            for j in range(self.__m):
                self.__normalized_data[j, i] = (self.__normalized_data[j, i] - self.__data_min_max[i][0]) \
                                               / (self.__data_min_max[i][1] - self.__data_min_max[i][0])
        s_cov = np.zeros((self.__n, self.__n))
        for i in range(self.__m):
            cur = self.__normalized_data[i, :].T * self.__normalized_data[i, :]
            s_cov += cur / self.__m
        eig_val, eig_vec = np.linalg.eig(s_cov)
        eig_ind = np.argsort(-eig_val)
        main_ind = eig_ind[: features]
        main_eig_vec = eig_vec[:, main_ind]
        self.__pca_data_mat = np.dot(self.__normalized_data, main_eig_vec)
        self.__pca_normalized_data = np.dot(self.__pca_data_mat, main_eig_vec.T)
        self.__pca_restored_data = np.zeros((self.__m, self.__n))
        for i in range(self.__m):
            for j in range(self.__n):
                self.__pca_restored_data[i][j] = self.__pca_normalized_data[i, j] \
                                                 * (self.__data_min_max[j][1] - self.__data_min_max[j][0]) \
                                                 + self.__data_min_max[j][0]
        # print(self.__pca_data_mat)
        # for x, y in zip(self.__normalized_data, self.__pca_normalized_data):
        #     print(f"{x} -> {y}")
        # for x, y in zip(self.__data, self.__pca_data):
        #     print(f"{x} => {y}")
        return self.__pca_data_mat
