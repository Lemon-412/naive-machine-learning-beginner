import numpy as np


class NaiveBayes:
    def __init__(self, training_x, training_y, is_continuous):
        """
        :param training_x: 训练集的特征集，离散变量必须已经离散化，下标从0开始
        :param training_y: 训练集的标签，必须已经离散化，下标从0开始
        :param is_continuous: 各特征是否是连续型， list of boolean
        """
        assert len(training_x) == len(training_y) != 0
        assert len(training_x[0]) == len(is_continuous) != 0

        self.__training_x = training_x
        self.__training_y = training_y
        self.__is_continuous = is_continuous

        self.__m = len(self.__training_x)  # 测试集样本数
        self.__n = len(self.__training_x[0])  # 特征维数

        self.__nx = None  # 各离散型特征的值域
        self.__ny = None  # 目标的值域

        self.__py = None  # 目标各值出现的频率
        self.__pyx = None  # 离散特征各值出现的条件概率
        # pyx[第i个目标分类][第j个特征][该特征的值v] = P(第j个特征值为v given 分类为i)

        self.__mean = None  # 各连续型特征的均值
        self.__variance = None  # 各连续型特征的方差

    def train(self):
        """
        计算先验概率和各条件概率，降低预测时间复杂度
        """
        self.__nx = []
        for i in range(self.__n):
            if self.__is_continuous[i]:
                self.__nx.append(0)  # 连续型变量不统计
            else:
                self.__nx.append(len(set(self.__training_x[:, i])))
        self.__ny = len(set(self.__training_y))

        self.__py = np.zeros(self.__ny)  # 初始值为0
        self.__pyx = [[np.ones(size) for size in self.__nx] for _ in range(self.__ny)]  # 初始值为1

        self.__mean = [np.zeros(self.__n) for _ in range(self.__ny)]  # 初始值为0
        self.__variance = [np.zeros(self.__n) for _ in range(self.__ny)]

        for x, y in zip(self.__training_x, self.__training_y):
            self.__py[y] += 1  # 先用于计数
            for i in range(self.__n):
                if self.__is_continuous[i]:
                    self.__mean[y] += 1  # 先用于计数
                else:
                    # print(f"y:{y} i:{i} x[i]:{x[i]} ")
                    self.__pyx[y][i][x[i]] += 1  # 先用于计数

        for y in range(self.__ny):
            for i in range(self.__n):
                if self.__is_continuous[i]:
                    self.__mean[y] /= self.__py[y]  # 转化为均值
                else:
                    self.__pyx[y][i] /= self.__py[y] + self.__nx[i]   # 后转化为修正后的频率

        for i in range(self.__n):
            if not self.__is_continuous[i]:
                continue
            for x, y in zip(self.__training_x.T[i], self.__training_y):
                self.__variance[y][i] += (x - self.__mean[y][i]) ** 2
            for y in range(self.__ny):
                self.__variance[y][i] /= self.__py[y]

        self.__py += 1  # 进行修正
        self.__py /= self.__m + self.__ny  # 后转化为修正后的频率

    def inference(self, inference_x):
        assert len(inference_x) == self.__n
        best_ans, best_val = None, None
        for y in range(self.__ny):
            cur_val = np.log(self.__py[y])
            for i in range(self.__n):
                if self.__is_continuous[i]:
                    cur_val -= (np.log(2 * np.pi) + self.__variance[y][i]) / 2
                    cur_val -= (self.__mean[y][i] - inference_x[i]) ** 2 / (2 * self.__variance[y][i])
                else:
                    cur_val += np.log(self.__pyx[y][i][inference_x[i]])
            if best_val is None or cur_val > best_val:
                best_val = cur_val
                best_ans = y
        return best_ans
