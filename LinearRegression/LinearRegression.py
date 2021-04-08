import torch
import numpy as np


class LinearRegression(torch.nn.Module):
    def __init__(self, training_x, training_y, lr=0.00001):
        super(LinearRegression, self).__init__()
        self.__training_x = torch.Tensor(training_x)
        self.__training_y = torch.Tensor(training_y)
        self.__linear = torch.nn.Linear(in_features=3, out_features=1, bias=True)
        self.__criterion = torch.nn.MSELoss(reduction='mean')
        self.__optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        y_pred = self.__linear(x)
        return y_pred

    def train(self, epoch_cnt=100):
        for epoch in range(epoch_cnt):
            y_pred = self(self.__training_x)
            loss = self.__criterion(y_pred, self.__training_y)
            # print(f"epoch: {epoch} loss: {loss.item()}")
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

    def inference(self, x):
        x = torch.Tensor(x)
        y = self(x)
        return y


if __name__ == '__main__':
    # linear_regression = LinearRegression(np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [4.0], [6.0]]))
    linear_regression = LinearRegression(
        np.array([[168, 51, 38], [179, 70, 42], [178, 74, 42], [155, 50, 36]]),
        np.array([[1], [0], [0], [1]])
    )
    linear_regression.train(epoch_cnt=500)
    y = linear_regression.inference([[179, 79, 44.5]])
    print(y)
