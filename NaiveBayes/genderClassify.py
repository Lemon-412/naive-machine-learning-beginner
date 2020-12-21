from NaiveBayes import NaiveBayes
import numpy as np

if __name__ == '__main__':
    raw_data = np.genfromtxt("student_data.csv", delimiter=",", skip_header=True)
    x = raw_data[:, 1:]
    y = np.array(raw_data.T[0].T, dtype=int)
    is_continuous = [True, True, True]
    naive_bayes = NaiveBayes(x, y, is_continuous)
    naive_bayes.train()
    try:
        while True:
            x = list(map(int, input("inference of height, weight, shoe size = ").split()))
            print(f"result={naive_bayes.inference(x)}\n")
    except EOFError:
        pass
