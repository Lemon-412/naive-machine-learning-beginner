from PCA import PCA
import numpy as np


def loadDataSet(file_name, delimiter=','):
    """
    加载数据集
    :param file_name: 数据集名称
    :param delimiter: 分隔符
    :return: 数据矩阵
    """
    with open(file_name, 'r', encoding='utf-8') as fr:
        string_arr = [line.strip().split(delimiter) for line in fr.readlines()]
        dat_arr = [list(map(float, line[1:])) for line in string_arr]
        label = [0 if line[0] == '男' else 1 for line in string_arr]
    return np.mat(dat_arr), label


def main():
    data, label = loadDataSet("data.csv")
    pca = PCA(data)
    lowDataMat, reconMat = pca.kl(2)
    print(len(data))
    print(len(reconMat))
    for x, y in zip(data, reconMat):
        print(f"{x} -> {y}")


if __name__ == '__main__':
    main()
