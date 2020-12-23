from NaiveBayes import NaiveBayes
import numpy as np

if __name__ == '__main__':
    raw_data = np.genfromtxt("student_data.csv", delimiter=",", skip_header=True)
    x = raw_data[:, 1:]
    y = np.array(raw_data.T[0].T, dtype=int)
    is_continuous = [True, True, True]

    cnt = 0
    for i in range(len(x)):
        train_x = np.delete(x, i, 0)
        train_y = np.delete(y, i, 0)
        test_x = x[i]
        test_y = y[i]
        naive_bayes = NaiveBayes(train_x, train_y, is_continuous)
        naive_bayes.train()
        test_result = naive_bayes.inference(test_x)
        print(f"{test_x}: {test_y} => {test_result}")
        if test_result == test_y:
            cnt += 1
    print("=============================")
    print(f"| result: {cnt / len(x) * 100}%")
    print("=============================")

    naive_bayes = NaiveBayes(x, y, is_continuous)
    naive_bayes.train()
    try:
        while True:
            x = list(map(int, input("inference of height, weight, shoe size = ").split()))
            print(f"result={naive_bayes.inference(x)}\n")
    except EOFError:
        pass
