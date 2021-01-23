from NaiveBayes import NaiveBayes
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from types import MethodType


def leave_one_out(x, y, is_continuous):
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
        naive_bayes = NaiveBayes(train_x, train_y, is_continuous)
        naive_bayes.train()
        test_result = naive_bayes.inference(test_x)
        if test_result == test_y:
            cnt += 1
    return cnt / len(x)


def inference(self, inference_x):
    """
    重写类内inference函数，用于绘制ROC曲线
    """
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
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    details = [
        (1, 3, 4),
        (2, 1, 3),
        (3, 1, 4),
    ]
    for feat, st, ed in details:
        x = np.array(raw_data.iloc[:, st: ed])
        is_continuous = [True] * feat
        acc = leave_one_out(x, y, is_continuous)
        print(f"accuracy using {feat} feature: {acc * 100}%")

    fig = make_subplots(
        rows=1, cols=2,
        column_width=[0.6, 0.4],
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "visualization of training data",
            "ROC curve of models using different features",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=np.array(raw_data).T[1],
            y=np.array(raw_data).T[2],
            z=np.array(raw_data).T[3],
            marker=dict(color=y), mode="markers",
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=2,
    )
    for i in range(1, 4):
        train_x = np.array(raw_data.iloc[:20, i: i + 1])
        train_y = np.array(np.array(raw_data.iloc[:20, :]).T[0].T, dtype=int)
        inference_x = np.array(raw_data.iloc[20:, i: i + 1])
        inference_y = np.array(np.array(raw_data.iloc[20:, :]).T[0].T, dtype=int)
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
        auc = 0.0
        plot_x, plot_y = [0], [0]
        for elem in ans:
            cnt[elem[1]] += 1
            plot_x.append(cnt[0] / tot[0])
            plot_y.append(cnt[1] / tot[1])
            if plot_x[-1] != plot_x[-2]:
                auc += (plot_x[-1] - plot_x[-2]) * plot_y[-1]
        fig.add_trace(
            go.Scatter(
                x=plot_x, y=plot_y,
                name=f"feature {i} (AUC={round(auc, 4)})",
                mode="lines",
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="False Positive Rate", range=[-0.01, 1.01], row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", range=[-0.01, 1.01], row=1, col=2)
    fig["layout"]["scene"]["xaxis"] = {"title": {"text": "height"}}
    fig["layout"]["scene"]["yaxis"] = {"title": {"text": "weight"}}
    fig["layout"]["scene"]["zaxis"] = {"title": {"text": "shoe size"}}
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
