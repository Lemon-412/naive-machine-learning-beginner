from copy import deepcopy
from math import log2


class ID3:
    def __init__(self, training_x, training_y, labels=None):
        """
        :param training_x: 训练集的特征集X^n数组
        :param training_y: 训练集的目标y^n数组
        :param labels: 各个特征的名称，默认值 [0, 1, 2, ..., n - 1]
        """
        assert len(training_x) == len(training_y) != 0
        if labels is None:
            labels = [_ for _ in range(len(training_x[0]))]
        assert len(training_x[0]) == len(labels)
        self.__training_x = training_x
        self.__training_y = training_y
        self.__labels = labels
        self.__tree = None

    def __str(self, cur, d):
        """
        递归打印决策树
        :param cur: 当前树节点
        :param d: 当前深度
        """
        if cur[0] == "_leaf":
            print(f"{d * ' '}\"{cur[1]}\"")
            return
        print(f"{d * ' '}{cur[0]}=?")
        d += 4
        for k, v in cur[1].items():
            print(f"{d * ' '}{k}:")
            self.__str(v, d + 4)
        d -= 4

    def __str__(self):
        if self.__tree is None:
            return ""
        self.__str(self.__tree, 0)
        return ""

    @property
    def tree(self):
        return self.__tree

    @staticmethod
    def __shannon_ent(y):
        """
        计算香农熵
        :param y: 包含目标y^n的数组
        :return: y^n的香农熵
        """
        ent = 0.0
        dic = dict()
        for elem in y:
            dic[elem] = dic.get(elem, 0) + 1
        for k, v in dic.items():
            prob = 1.0 * v / len(y)
            ent -= prob * log2(prob)
        return ent

    @staticmethod
    def __find_best_feature(x, y):
        """
        :param x: 特征集X^m数组
        :param y: 目标y^m数组
        :return: 信息增益最大的feature下标
        """
        x = deepcopy(x)
        y = deepcopy(y)
        best_feature = -1
        ent = ID3.__shannon_ent(y)
        best_gain = 0.0
        for i in range(len(x[0])):  # 尝试分裂各个feature，计算信息增益
            dic = dict()
            for cx, cy in zip(x, y):  # 将去除该x的(X,y)按x值分类存入字典
                if dic.get(cx[i], None) is None:
                    dic[cx[i]] = [cy]
                else:
                    dic[cx[i]].append(cy)
            gain = ent  # 信息增益 = 原始香农熵 - \sigma{各个分裂子集的香农熵}
            for elem in dic.values():
                gain -= 1.0 * len(elem) / len(y) * ID3.__shannon_ent(elem)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
        return best_feature

    @staticmethod
    def __generate_tree(x, y, labels):
        """
        递归生成决策树
        :param x: 特征集X^m数组
        :param y: 目标y^m数组
        :param labels: 特征名
        :return: 决策子树
        """
        x = deepcopy(x)
        y = deepcopy(y)
        labels = deepcopy(labels)

        if y.count(y[0]) == len(y):  # 所有y都是一类，无需分类
            return ["_leaf", y[0]]

        if x.count(x[0]) == len(x):  # 所有x相同，无法分类(或没有标签可分)
            dic = dict()
            for elem in y:
                dic[elem] = dic.get(elem, 0) + 1
            cnt, val = 0, ""
            for k, v in dic.items():  # 选择出现频率最高的一项作为叶节点值
                if v > cnt:
                    cnt, val = v, k
            return ["_leaf", val]

        best_id = ID3.__find_best_feature(x, y)  # 寻找信息增益最大的feature下标
        tree = [labels[best_id], dict()]
        del labels[best_id]
        dic = dict()
        for cx, cy in zip(x, y):  # 按照信息增益最大的feature划分，递归形成决策树
            v = cx[best_id]
            del cx[best_id]
            if dic.get(v, None) is None:
                dic[v] = {"x": [cx], "y": [cy]}
            else:
                dic[v]["x"].append(cx)
                dic[v]["y"].append(cy)
        for k, v in dic.items():
            tree[1][k] = ID3.__generate_tree(v["x"], v["y"], labels)
        return tree

    def generate_tree(self):
        self.__tree = ID3.__generate_tree(self.__training_x, self.__training_y, self.__labels)

    def inference(self, x):
        """
        使用决策树进行预测
        :param x: 一个特征数组
        :return: 预测值
        """
        assert len(x) == len(self.__labels)
        dic = dict(zip(self.__labels, x))
        cur = self.__tree
        while cur[0] != "_leaf":
            val = dic[cur[0]]  # 此处如出现未知特征值会预测失败
            cur = cur[1][val]
        return cur[1]
