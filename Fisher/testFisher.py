import numpy as np
import pandas as pd
from Fisher import Fisher
import plotly.graph_objects as go
from types import MethodType


def cross_validation(x, y):
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
        fisher = Fisher(train_x, train_y)
        fisher.train()
        test_result = fisher.inference(test_x)
        if test_result == test_y:
            cnt += 1
    return cnt / len(x)


def inference(self, inference_x):
    return np.dot(self._Fisher__w.T, np.array([inference_x]).T)


def main():
    raw_data = pd.read_csv("student_data.csv")
    x = np.array(raw_data.iloc[:, 3:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    # print(f"x: {x}")
    # print(f"y: {y}")
    fisher = Fisher(x, y)
    fisher.train()
    # return

    input("==============================================")
    acc = cross_validation(x, y)
    print(f"accuracy: {acc * 100}%")

    input("==============================================")
    train_x = np.array(raw_data.iloc[:25, 3:])
    train_y = np.array(np.array(raw_data.iloc[:25, :]).T[0].T, dtype=int)
    inference_x = np.array(raw_data.iloc[20:, 3:])
    inference_y = np.array(np.array(raw_data.iloc[25:, :]).T[0].T, dtype=int)
    fisher = Fisher(train_x, train_y)
    fisher.inference = MethodType(inference, fisher)
    fisher.train()
    ans = []
    tot, cnt = [0, 0], [0, 0]
    for x_i, y_i in zip(inference_x, inference_y):
        ans.append([fisher.inference(x_i), y_i])
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
    fig = go.Figure()
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, y0=0, y1=1,
    )
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode="lines",
    ))
    fig.update_xaxes(constrain='domain')
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"ROC curve of Fisher classifier (AUC={auc})"
    )
    fig.show()


if __name__ == '__main__':
    main()
