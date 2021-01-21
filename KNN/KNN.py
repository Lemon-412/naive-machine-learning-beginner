import numpy as np
from copy import deepcopy
from random import randint


class KNN:
    def __init__(self, train_x, train_y, k):
        """
        kNN初始化函数
        :param train_x: numpy.array，训练集的特征
        :param train_y: numpy.array，训练集的目标
        :param k: int 预测时选取的临近样本个数
        """
        assert(len(train_x) == len(train_y))
        self.__n = len(train_x)
        self.__k = k
        self.__train_x = train_x
        self.__train_y = train_y

    def train(self):
        return

    def inference(self, inference_x):
        """
        kNN预测函数
        :param inference_x: numpy.array，预测特征
        :return: 预测结果
        """
        tmp = []
        for i in range(self.__n):  # 对每一个train计算曼哈顿距离
            tmp.append(np.array([np.linalg.norm(inference_x - self.__train_x[i]), self.__train_y[i]]))
        tmp = np.array(tmp)
        tmp = tmp[np.argsort(tmp[:, 0])][0:self.__k]  # 根据第一个元素排序取前k大
        ans = np.argmax(np.bincount(np.array(tmp.T[1], dtype=np.int32)))  # 找到频数最多的数
        return ans


class CondenseKNN:
    def __init__(self, train_x, train_y, k):
        """
        Condense kNN初始化函数
        :param train_x: numpy.array，训练集的特征
        :param train_y: numpy.array，训练集的目标
        :param k: int 预测时选取的临近样本个数
        """
        assert(len(train_x) == len(train_y))
        self.__n = len(train_x)
        self.__k = k
        self.__raw_x = train_x
        self.__raw_y = train_y
        self.__store_x = None
        self.__store_y = None
        self.__grab_bag_x = None
        self.__grab_bag_y = None

    def train(self):
        self.__store_x = []
        self.__store_y = []
        self.__grab_bag_x = deepcopy(self.__raw_x)
        self.__grab_bag_y = deepcopy(self.__raw_y)
        for _ in range(self.__k):
            assert len(self.__grab_bag_x) == len(self.__grab_bag_y)
            cur = randint(0, len(self.__grab_bag_x) - 1)
            self.__store_x.append(self.__grab_bag_x[cur])
            self.__store_y.append(self.__grab_bag_y[cur])
            self.__grab_bag_x = np.delete(self.__grab_bag_x, cur, axis=0)
            self.__grab_bag_y = np.delete(self.__grab_bag_y, cur)
        while True:
            knn = KNN(self.__store_x, self.__store_y, self.__k)
            assert len(self.__grab_bag_x) == len(self.__grab_bag_y)
            for i in range(len(self.__grab_bag_x)):
                if knn.inference(self.__grab_bag_x[i]) != self.__grab_bag_y[i]:
                    self.__store_x.append(self.__grab_bag_x[i])
                    self.__store_y.append(self.__grab_bag_y[i])
                    self.__grab_bag_x = np.delete(self.__grab_bag_x, i, axis=0)
                    self.__grab_bag_y = np.delete(self.__grab_bag_y, i)
                    break
            else:
                break
        # print(f"{len(self.__raw_x)} -> {len(self.__store_x)}")

    def inference(self, inference_x):
        """
        Condense kNN预测函数
        :param inference_x: numpy.array，预测特征
        :return: 预测结果
        """
        tmp = []
        for i in range(len(self.__store_x)):  # 对每一个train计算曼哈顿距离
            tmp.append(np.array([np.linalg.norm(inference_x - self.__store_x[i]), self.__store_y[i]]))
        tmp = np.array(tmp)
        tmp = tmp[np.argsort(tmp[:, 0])][0:self.__k]  # 根据第一个元素排序取前k大
        ans = np.argmax(np.bincount(np.array(tmp.T[1], dtype=np.int32)))  # 找到频数最多的数
        return ans
