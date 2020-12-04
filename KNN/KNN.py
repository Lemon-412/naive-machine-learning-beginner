import numpy as np


class KNN:
    def __init__(self, train_x, train_y, k):
        assert(len(train_x) == len(train_y))
        self.__n = len(train_x)
        self.__k = k
        self.__train_x = train_x
        self.__train_y = train_y

    def inference(self, inference_x):
        tmp = []
        for i in range(self.__n):  # 对每一个train计算曼哈顿距离
            tmp.append(np.array([np.linalg.norm(inference_x - self.__train_x[i]), self.__train_y[i]]))
        tmp = np.array(tmp)
        tmp = tmp[np.argsort(tmp[:, 0])][0:self.__k]  # 根据第一个元素排序取前k大
        tmp = np.argmax(np.bincount(np.array(tmp.T[1], dtype=np.int32)))  # 找到频数最多的数
        return tmp

