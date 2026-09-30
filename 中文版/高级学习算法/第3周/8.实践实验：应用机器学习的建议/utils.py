import numpy as np
from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 我们只取前两个特征。
    y = iris.target

    X = X[y != 2] # 只有两个类别
    y = y[y != 2]
    return X, y