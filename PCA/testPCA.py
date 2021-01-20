import sys
from PCA import PCA
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.append("..")
from NaiveBayes.NaiveBayes import NaiveBayes


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


def main():
    raw_data = pd.read_csv("student_data.csv")
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)

    fig = make_subplots(
        rows=2, cols=2,
        column_width=[0.5, 0.5],
        row_width=[0.5, 0.5],
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}],
            [{"type": "scatter3d"}, {"type": "scatter"}]
        ]
    )

    energy = [0]
    for i in range(1, 4, 1):
        pca = PCA(x)
        pca.pca(features=i)
        energy.append(pca.energy)
        fig.add_scatter3d(
            x=np.array(pca.restored_data).T[0],
            y=np.array(pca.restored_data).T[1],
            z=np.array(pca.restored_data).T[2],
            row=(i + 1) // 2, col=2 - i % 2,
            name=f"{i} features", mode="markers",
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(0, 4, 1)), y=energy,
            name=f"energy",
            mode="lines",
        ),
        row=2, col=2,
    )
    # fig.update_layout(template="plotly_dark")
    fig.show()

    print("Training with NaiveBayes...")
    accuracy = leave_one_out(x, y, [True, True, True])
    print(f"accuracy (3): {accuracy * 100}%")
    accuracy = leave_one_out(np.array(raw_data.iloc[:, 1:3]), y, [True, True])
    print(f"accuracy (2): {accuracy * 100}%")
    accuracy = leave_one_out(np.array(raw_data.iloc[:, 2:3]), y, [True])
    print(f"accuracy (1): {accuracy * 100}%")
    pca = PCA(x)
    pca_x = np.array(pca.pca(features=2))
    accuracy = leave_one_out(pca_x, y, [True, True])
    print(f"accuracy (pca -> 2): {accuracy * 100}%")
    pca = PCA(x)
    pca_x = np.array(pca.pca(features=1))
    accuracy = leave_one_out(pca_x, y, [True])
    print(f"accuracy (pca -> 1): {accuracy * 100}%")


if __name__ == '__main__':
    main()
