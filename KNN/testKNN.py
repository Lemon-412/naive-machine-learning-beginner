from KNN import KNN
import numpy as np
from os import listdir

dataset_dir = "datasets/trainingDigits"
inference_dir = "datasets/testDigits"

if __name__ == '__main__':
    training_x = []
    training_y = []
    inference_x = []
    inference_y = []

    for file_name in listdir(dataset_dir):
        with open(dataset_dir + "/" + file_name, "r") as FILE:
            training_x.append(np.array(list(map(int, FILE.read().replace("\n", "")))))
        training_y.append(int(file_name.split("_")[0]))

    for file_name in listdir(inference_dir):
        with open(inference_dir + "/" + file_name, "r") as FILE:
            inference_x.append(np.array(list(map(int, FILE.read().replace("\n", "")))))
        inference_y.append(int(file_name.split("_")[0]))

    training_x = np.array(training_x)
    training_y = np.array(training_y)
    inference_x = np.array(inference_x)
    inference_y = np.array(inference_y)
    result = []

    knn = KNN(training_x, training_y, 10)
    err = 0
    for i in range(len(inference_x)):
        ans = knn.inference(inference_x[i])
        result.append(ans)
        if inference_y[i] != ans:
            print(f"inference {i}: real={inference_y[i]} -> ans={ans}")
            err += 1
    print(f"accuracy: {1.0 - err / len(inference_x)}")

