from PCA import PCA
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.express.colors as pc


def main():
    raw_data = pd.read_csv("student_data.csv")
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(np.array(raw_data).T[0].T, dtype=int)
    pca = PCA(x)
    pca_x = pca.pca(features=1)
    err = pca.restored_data - x


if __name__ == '__main__':
    main()
