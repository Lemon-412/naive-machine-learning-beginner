import numpy as np
import pandas as pd
from Fisher import Fisher
import plotly.graph_objects as go
import plotly.express as px
from types import MethodType


def leave_one_out(x, y):
    """
    使用留一法对模型进行评估
    :param x: 数据集特征
    :param y: 数据集标签
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


def hold_out(x, y, scale=0.7):
    """
    使用留出法对模型进行评估
    :param x: 数据集特征
    :param y: 数据集标签
    :param scale: 训练集占比
    :return: 模型留出法下的正确率
    """
    n = int(len(y) * scale)
    fisher = Fisher(x[:n], y[:n])
    fisher.train()
    cnt = 0
    for i in range(n, len(y)):
        test_result = fisher.inference(x[i])
        if test_result == y[i]:
            cnt += 1
    return cnt / (len(y) - n)


def inference(self, inference_x):
    return np.dot(self._Fisher__w.T, np.array([inference_x]).T)


def calculate_roc(train_x, train_y, inference_x, inference_y):
    """
    计算给定数据集，绘制ROC曲线并计算AUC
    :return: AUC
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
    fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="lines"))
    fig.update_xaxes(constrain='domain')
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"ROC curve of Fisher classifier (AUC={auc})"
    )
    fig.show()
    return auc


def main():
    raw_data = pd.read_csv("student_data.csv")

    for feature in range(1, 4, 1):
        print(f"================== {feature} feature ==================")
        x = np.array(raw_data.iloc[:, 4 - feature:])
        y = np.array(np.array(raw_data).T[0].T, dtype=int)
        print(f"accuracy using leave_one_out: {leave_one_out(x, y) * 100}%")
        print(f"accuracy using hold_out: {hold_out(x, y, scale=0.7) * 100}%")

    print("===========================================")
    train_x = np.array(raw_data.iloc[:10, 1:])
    train_y = np.array(np.array(raw_data.iloc[:10, :]).T[0].T, dtype=int)
    inference_x = np.array(raw_data.iloc[10:, 1:])
    inference_y = np.array(np.array(raw_data.iloc[10:, :]).T[0].T, dtype=int)
    auc = calculate_roc(train_x, train_y, inference_x, inference_y)
    print(f"auc: {auc}")

    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    fisher = Fisher(x, y)
    fisher.train()
    print(f"w={fisher.w.T}")
    print(f"w*={fisher.w0}")

    dense = 200
    x_space = np.linspace(np.array(raw_data).T[1].T.min(), np.array(raw_data).T[1].T.max(), dense)
    y_space = np.linspace(np.array(raw_data).T[2].T.min(), np.array(raw_data).T[2].T.max(), dense)
    bayes_space = np.zeros((dense, dense))
    fisher_space = np.zeros((dense, dense))
    np.seterr(invalid='ignore')
    for i in range(dense):
        for j in range(dense):
            bayes_space[i][j] = 29.0046570730851 + 44.3079999492738 * np.sqrt(
                -4.39895231808396 * 10 ** (-5) * x_space[i] ** 2
                + 0.013276593367934 * x_space[i]
                - 3.8402205477989 * 10 ** (-5) * y_space[j] ** 2
                + 0.003505508996134 * y_space[j] - 1
            )
            fisher_space[i][j] = -1 / fisher.w[2] * (fisher.w0 + fisher.w[0] * x_space[i] + fisher.w[1] * y_space[j])
    fig = px.scatter_3d(raw_data, x="height", y="weight", z="shoe_size", color="gender")
    fig.add_trace(go.Surface(x=x_space, y=y_space, z=bayes_space, opacity=0.50, showscale=False))
    fig.add_trace(go.Surface(x=x_space, y=y_space, z=fisher_space, opacity=0.50, showscale=False))
    fig.show()


if __name__ == '__main__':
    main()
