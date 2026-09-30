# C2_W1 工具
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 绘制多类别训练点
def plot_mc_data(X, y, class_labels=None, legend=False,size=40):
    classes = np.unique(y)
    for i in classes:
        label = class_labels[i] if class_labels else "class {}".format(i)
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],  cmap=plt.cm.Paired,
                    edgecolor='black', s=size, label=label)
    if legend: plt.legend()
        

#绘制多类别分类决策边界
# 这个版本处理非向量化预测（对各点增加了 for 循环）
def plot_cat_decision_boundary(X,predict , class_labels=None, legend=False, vector=True):

    # 创建用于绘图的网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max(x_max-x_min, y_max-y_min)/200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]

    #为网格中的每个点做预测
    if vector:
        Z = predict(points)
    else:
        Z = np.zeros((len(points),))
        for i in range(len(points)):
            Z[i] = predict(points[i].reshape(1,2))
    Z = Z.reshape(xx.shape)

    #等高线图突出显示数值之间的边界——本例中即类别之间的边界
    plt.figure()
    plt.contour(xx, yy, Z, colors='g') 
    plt.axis('tight')