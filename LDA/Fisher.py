import numpy as np


class Fisher:
    def __init__(self, training_x, training_y):
        """
        :param training_x: np.array类型，训练用数据集
        :param training_y: np.array类型，训练用标签，须为0, 1二分类
        """
        assert len(training_x) == len(training_y)
        self.__training_x = training_x
        self.__training_y = training_y
        self.__n = len(self.__training_x[0])
        self.__w = None
        self.__w0 = None

    @property
    def w(self):
        return self.__w

    @property
    def w0(self):
        return self.__w0

    def train(self):
        m = np.zeros((2, self.__n))
        s = np.zeros((self.__n, self.__n))
        cnt = [0, 0]
        for x, y in zip(self.__training_x, self.__training_y):
            cnt[y] += 1
            m[y] += x
        m[0] /= cnt[0]
        m[1] /= cnt[1]
        for x, y in zip(self.__training_x, self.__training_y):
            cur = np.array([x - m[y]])
            s += np.matmul(cur.T, cur)
        self.__w = np.matmul(np.linalg.inv(s), np.array([m[0] - m[1]]).T)
        self.__w0 = -np.matmul(np.matmul(m[0] + m[1], np.linalg.inv(s)), np.array([m[0] - m[1]]).T) / 2 \
                    - np.log(cnt[1] / cnt[0]) / (cnt[0] + cnt[1] - 2)

    def inference(self, inference_x):
        assert len(inference_x) == self.__n
        inference_y = np.matmul(self.__w.T, np.array([inference_x]).T) + self.__w0
        return 0 if inference_y > 0 else 1
