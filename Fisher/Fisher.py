import numpy as np


class Fisher:
    def __init__(self, training_x, training_y):
        assert len(training_x) == len(training_y)
        self.__training_x = training_x
        self.__training_y = training_y
        self.__n = len(self.__training_x[0])
        self.__w = None
        self.__w0 = None

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
            s += np.dot(cur.T, cur)
        self.__w = np.dot(np.linalg.inv(s), np.array([m[0] - m[1]]).T)
        self.__w0 = -np.dot(np.dot(m[0] + m[1], np.linalg.inv(s)), np.array([m[0] - m[1]]).T) / 2 - np.log(cnt[1] / cnt[0])

    def inference(self, inference_x):
        inference_y = np.dot(self.__w.T, np.array([inference_x]).T) + self.__w0
        return 0 if inference_y > 0 else 1
