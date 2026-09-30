import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """计算 z 的 sigmoid

    参数
    ----------
    z : array_like
        标量或任意大小的 numpy 数组。

    返回
    -------
     g : array_like
         sigmoid(z)"""
    # （约 1 行代码）
    # g = 
    ### 在此处编写代码 ###
    g = 1/(1+np.exp(-z))
    ### 代码编写结束 ###
    
    return g

def plot_data(X, y):
    
    # 找出正例与负例的索引
    pos = y == 1
    neg = y == 0
    
    # 绘制样本
    plt.plot(X[pos, 0], X[pos, 1], 'r+', label="y=1")
    plt.plot(X[neg, 0], X[neg, 1], 'ko', label="y=0")