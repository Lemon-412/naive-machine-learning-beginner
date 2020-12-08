from copy import deepcopy
from math import log2


class ID3:
    def __init__(self, training_x, training_y, labels=None):
        assert len(training_x) == len(training_y) != 0
        if labels is None:
            labels = [_ for _ in range(len(training_x[0]))]
        assert len(training_x[0]) == len(labels)
        self.__training_x = training_x
        self.__training_y = training_y
        self.__labels = labels
        self.__tree = None

    def __str(self, cur, d):
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
        x = deepcopy(x)
        y = deepcopy(y)
        best_feature = -1
        ent = ID3.__shannon_ent(y)
        best_gain = 0.0
        for i in range(len(x[0])):  # 尝试分裂各个feature，计算信息增益
            dic = dict()
            for cx, cy in zip(x, y):
                if dic.get(cx[i], None) is None:
                    dic[cx[i]] = [cy]
                else:
                    dic[cx[i]].append(cy)
            gain = ent
            for elem in dic.values():
                gain -= 1.0 * len(elem) / len(y) * ID3.__shannon_ent(elem)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
        return best_feature

    @staticmethod
    def __generate_tree(x, y, labels):
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
            for k, v in dic.items():
                if v > cnt:
                    cnt, val = v, k
            return ["_leaf", val]

        best_id = ID3.__find_best_feature(x, y)
        tree = [labels[best_id], dict()]
        del labels[best_id]
        dic = dict()
        for cx, cy in zip(x, y):
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
        assert len(x) == len(self.__labels)
        dic = dict(zip(self.__labels, x))
        cur = self.__tree
        while cur[0] != "_leaf":
            val = dic[cur[0]]
            cur = cur[1][val]
        return cur[1]
