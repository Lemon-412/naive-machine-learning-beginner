import numpy as np
import pandas as pd
from Fisher import Fisher
import plotly.graph_objects as go
from types import MethodType


def cross_validation(x, y):
    """
    使用留一法验证法对模型进行评估
    :return: 模型留一法下的正确率
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


def calculate_roc(train_x, train_y, inference_x, inference_y):
    """
    计算给定数据集，绘制ROC曲线并计算AUC
    """
    fisher = Fisher(train_x, train_y)
    fisher.inference = MethodType(inference, fisher)  # 重写inference函数用于计算ROC
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
    fig.add_trace(go.Scatter(x=plot_x, y=plot_y,mode="lines",))
    fig.update_xaxes(constrain='domain')
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"ROC curve of Fisher classifier (AUC={auc})"
    )
    fig.show()


def main():
    raw_data = pd.read_csv("student_data.csv")

    print("==============================================")
    x = np.array(raw_data.iloc[:, 1:3])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    acc = cross_validation(x, y)
    print(f"accuracy: {acc * 100}%")

    print("==============================================")
    train_x = np.array(raw_data.iloc[:10, 1:])
    train_y = np.array(np.array(raw_data.iloc[:10, :]).T[0].T, dtype=int)
    inference_x = np.array(raw_data.iloc[10:, 1:])
    inference_y = np.array(np.array(raw_data.iloc[10:, :]).T[0].T, dtype=int)
    calculate_roc(train_x, train_y, inference_x, inference_y)


if __name__ == '__main__':
    main()
