from KNN import KNN, CondenseKNN
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from copy import deepcopy
import plotly.graph_objects as go
import sys
from types import MethodType

sys.path.append("..")
from PCA.PCA import PCA


def leave_one_out(x, y, k, condense=False):
    """
    使用留一法验证法对模型进行评估
    :return: 模型留一法验证下的正确率
    """
    cnt = 0
    for i in range(len(x)):
        train_x = np.delete(x, i, 0)
        train_y = np.delete(y, i, 0)
        test_x = x[i]
        test_y = y[i]
        if condense:
            knn = CondenseKNN(train_x, train_y, k)
        else:
            knn = KNN(train_x, train_y, k)
        knn.train()
        test_result = knn.inference(test_x)
        if test_result == test_y:
            cnt += 1
    return cnt / len(x)


def normalize(data):
    """
    将数据进行归一化处理
    :param data: 待处理的数据，np.array
    :return: 归一化后的数据
    """
    data = deepcopy(data).T
    for i in range(len(data)):
        _min, _max = np.min(data[i]), np.max(data[i])
        data[i] = (data[i] - _min) / (_max - _min)
    return deepcopy(data.T)


def inference(self, inference_x):
    """
    对二分类问题重写KNN预测函数，得到决策可信度
    """
    tmp = []
    for i in range(self._KNN__n):  # 对每一个train计算曼哈顿距离
        tmp.append(np.array([np.linalg.norm(inference_x - self._KNN__train_x[i]), self._KNN__train_y[i]]))
    tmp = np.array(tmp)
    tmp = tmp[np.argsort(tmp[:, 0])][0:self._KNN__k]  # 根据第一个元素排序取前k大
    return sum(tmp.T[1]) / self._KNN__k


def main():
    raw_data = pd.read_csv("student_data.csv")
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    normalized_x = normalize(x)
    pca = PCA(x)
    pca_x = np.array(pca.pca(features=2))

    fig = make_subplots(
        rows=1, cols=2,
        column_width=[0.5, 0.5],
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "kNN/condense kNN performance of different training sets",
            "decision boundary of kNN using PCA data (2 dimensions)",
        )
    )

    details = [
        (x, False, "kNN + raw data"),
        (normalized_x, False, "kNN + normalized data"),
        (pca_x, False, "kNN + PCA (2 dimensions)"),
        (normalized_x, True, "condense kNN + normalized data"),
        (pca_x, True, "condense kNN + PCA (2 dimensions)"),
    ]
    for x_data, is_condense, name in details:
        accuracy = []
        for k in range(1, 50, 2):
            accuracy.append(leave_one_out(x_data, y, k, condense=is_condense))
        fig.add_trace(
            go.Scatter(
                x=list(range(1, 50, 2)), y=accuracy,
                name=name, mode="lines",
            ),
            row=1, col=1,
        )

    knn = KNN(pca_x, y, 15)
    knn.inference = MethodType(inference, knn)  # 重写inference函数
    x_range = np.arange(pca_x[:, 0].min() - 0.1, pca_x[:, 0].max() + 0.1, 0.01)
    y_range = np.arange(pca_x[:, 1].min() - 0.1, pca_x[:, 1].max() + 0.1, 0.01)
    print(f"x_range: {len(x_range)}, y_range: {len(y_range)}")
    zz = []
    for _x in x_range:
        zz.append([])
        for _y in y_range:
            ret = knn.inference(np.array([_x, _y]))
            zz[-1].append(ret)
    zz = np.array(zz).T

    fig.add_trace(
        go.Scatter(
            x=pca_x[y == 0, 0], y=pca_x[y == 0, 1],
            name="male data", mode="markers", marker_symbol="square",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=pca_x[y == 1, 0], y=pca_x[y == 1, 1],
            name="female data", mode="markers", marker_symbol="circle",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Contour(
            x=x_range, y=y_range, z=zz,
            showscale=False, colorscale='RdBu', opacity=0.4,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="accuracy using leave one out", range=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="PCA training set dimension 1", row=1, col=2)
    fig.update_yaxes(title_text="PCA training set dimension 2", row=1, col=2)
    fig.update_layout(title="Performance of kNN and Visualization of kNN Decision Boundary")
    fig.show()

    knn = CondenseKNN(x, y, 5)
    knn.train()
    print(f"condensed kNN data set size: {len(x)} -> {len(knn.condense_set[0])}")


if __name__ == '__main__':
    main()
