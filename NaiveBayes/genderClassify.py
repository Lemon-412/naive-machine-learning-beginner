from NaiveBayes import NaiveBayes
import plotly.express as px
import numpy as np
import pandas as pd
from types import MethodType


def cross_validation(x, y, is_continuous):
    """
    使用交叉验证法对模型进行评估
    :return: 模型交叉验证下的正确率
    """
    cnt = 0
    for i in range(len(x)):
        train_x = np.delete(x, i, 0)
        train_y = np.delete(y, i, 0)
        test_x = x[i]
        test_y = y[i]
        naive_bayes = NaiveBayes(train_x, train_y, is_continuous)
        naive_bayes.train()
        test_result = naive_bayes.inference(test_x)
        if test_result == test_y:
            cnt += 1
    return cnt / len(x)


def inference(self, inference_x):
    assert len(inference_x) == self._NaiveBayes__n
    val_0 = np.log(self._NaiveBayes__py[0])
    for i in range(self._NaiveBayes__n):
        if self._NaiveBayes__is_continuous[i]:
            val_0 -= (np.log(2 * np.pi) + np.log(self._NaiveBayes__variance[0][i])) / 2
            val_0 -= (self._NaiveBayes__mean[0][i] - inference_x[i]) ** 2 / (2 * self._NaiveBayes__variance[0][i])
        else:
            val_0 += np.log(self._NaiveBayes__pyx[0][i][inference_x[i]])
    val_1 = np.log(self._NaiveBayes__py[1])
    for i in range(self._NaiveBayes__n):
        if self._NaiveBayes__is_continuous[i]:
            val_1 -= (np.log(2 * np.pi) + np.log(self._NaiveBayes__variance[1][i])) / 2
            val_1 -= (self._NaiveBayes__mean[1][i] - inference_x[i]) ** 2 / (2 * self._NaiveBayes__variance[1][i])
        else:
            val_1 += np.log(self._NaiveBayes__pyx[1][i][inference_x[i]])
    return val_0 - val_1


def main():
    raw_data = pd.read_csv("student_data.csv")
    fig = px.scatter_3d(raw_data, x="height", y="weight", z="shoe_size", color="gender")
    fig.show()

    print("=================================================")
    x = np.array(raw_data.iloc[:, 3:4])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    is_continuous = [True]
    acc = cross_validation(x, y, is_continuous)
    print(f"accuracy using 1 feature: {acc * 100}%")

    x = np.array(raw_data.iloc[:, 1:3])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    is_continuous = [True, True]
    acc = cross_validation(x, y, is_continuous)
    print(f"accuracy using 2 features: {acc * 100}%")

    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    is_continuous = [True, True, True]
    acc = cross_validation(x, y, is_continuous)
    print(f"accuracy using 3 features: {acc * 100}%")

    print("=================================================")
    train_x = np.array(raw_data.iloc[:4, 2:3])
    train_y = np.array(np.array(raw_data.iloc[:4, :]).T[0].T, dtype=int)
    inference_x = np.array(raw_data.iloc[4:, 2:3])
    inference_y = np.array(np.array(raw_data.iloc[4:, :]).T[0].T, dtype=int)
    is_continuous = [True]
    naive_bayes = NaiveBayes(train_x, train_y, is_continuous)
    naive_bayes.inference = MethodType(inference, naive_bayes)
    naive_bayes.train()
    ans = []
    tot, cnt = [0, 0], [0, 0]
    for x_i, y_i in zip(inference_x, inference_y):
        ans.append([naive_bayes.inference(x_i), y_i])
        tot[y_i] += 1
    ans.sort()
    plot_x, plot_y = [0], [0]
    for elem in ans:
        cnt[elem[1]] += 1
        plot_x.append(cnt[0] / tot[0])
        plot_y.append(cnt[1] / tot[1])
    fig = px.area(
        x=plot_x, y=plot_y,
        title="ROC curve",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
    )
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, y0=0, y1=1,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()

    # is_continuous = [True, True, True]
    # naive_bayes = NaiveBayes(x, y, is_continuous)
    # try:
    #     while True:
    #         x = list(map(int, input("inference of height, weight, shoe size = ").split()))
    #         print(f"result={naive_bayes.inference(x)}\n")
    # except EOFError:
    #     pass


if __name__ == '__main__':
    main()
