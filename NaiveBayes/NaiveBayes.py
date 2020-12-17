import numpy as np


class NaiveBayes:
    def __init__(self, training_x, training_y):
        """
        :param training_x: 训练集的特征集，离散变量必须已经离散化，下标从0开始
        :param training_y: 训练集的标签，必须已经离散化，下标从0开始
        """
        assert len(training_x) == len(training_y) != 0
        self.__training_x = training_x
        self.__training_y = training_y
        self.__m = len(self.__training_x)  # 测试集样本数
        self.__n = len(self.__training_x[0])  # 特征维数
        self.__nx = None  # 各特征值域
        self.__ny = None  # 目标值域
        self.__py = None  # 目标各值的概率
        self.__pyx = None  # pyx[第i个目标分类][第j个特征][该特征的值v] = P(第j个特征值为v given 分类为i)

    def train(self):
        """
        计算先验概率和各条件概率，降低预测时间复杂度
        """
        self.__nx = np.array([len(set(self.__training_x[:, i])) for i in range(self.__n)])
        self.__ny = len(set(self.__training_y))

        self.__py = np.zeros(self.__ny)  # 初始值为0
        self.__pyx = [[np.ones(size) for size in self.__nx] for _ in range(self.__ny)]  # 初始值为1
        for x, y in zip(self.__training_x, self.__training_y):
            self.__py[y] += 1  # 计数
            for i in range(self.__n):
                self.__pyx[y][i][x[i]] += 1  # 计数
        for y in range(self.__ny):
            for i in range(self.__n):
                self.__pyx[y][i] /= self.__py[y] + self.__nx[i]   # 转化为修正后的频率
        self.__py += 1  # 修正
        self.__py /= self.__m + self.__ny  # 转化为修正后的频率

    def inference(self, inference_x):
        assert len(inference_x) == self.__n
        best_ans, best_val = None, None
        for y in range(self.__ny):
            cur_val = np.log(self.__py[y])
            for i in range(self.__n):
                cur_val += np.log(self.__pyx[y][i][inference_x[i]])
            if best_val is None or cur_val > best_val:
                best_val = cur_val
                best_ans = y
        return best_ans
