from LinearRegression import LinearRegression
import numpy as np
import pandas as pd


def leave_one_out(x, y):
    for i in range(len(x)):
        train_x = np.delete(x, i, 0)
        train_y = np.delete(y, i, 0)
        test_x = x[i]
        test_y = y[i]
        linear_regression = LinearRegression(train_x, train_y, lr=0.00001)
        linear_regression.train(epoch_cnt=10000)
        test_result = linear_regression.inference([test_x])
        print(f"{test_y} => {test_result.item()}")


def main():
    raw_data = pd.read_csv("student_data.csv")
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(raw_data.iloc[:, :1], dtype=int)
    leave_one_out(x, y)
    linear_regression = LinearRegression(x, y, lr=0.00001)
    linear_regression.train(epoch_cnt=10000)
    print(linear_regression.inference([[168, 51, 38]]).item())
    print(linear_regression.inference([[179, 70, 42]]).item())


if __name__ == '__main__':
    main()
