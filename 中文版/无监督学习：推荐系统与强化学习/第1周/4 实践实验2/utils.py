import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("数据/X_part1.npy")
    X_val = np.load("数据/X_val_part1.npy")
    y_val = np.load("数据/y_val_part1.npy")
    return X, X_val, y_val

def load_data_multi():
    X = np.load("数据/X_part2.npy")
    X_val = np.load("数据/X_val_part2.npy")
    y_val = np.load("数据/y_val_part2.npy")
    return X, X_val, y_val


def multivariate_gaussian(X, mu, var):
    """计算样本 X 在参数为 mu 和 var 的多元高斯分布下的概率密度函数。
   如果 var 是矩阵，则将其视为协方差矩阵。如果 var 是向量，
   则将其视为各维度方差取值组成的向量（对角协方差矩阵"""
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p
        
def visualize_fit(X, mu, var):
    """这个可视化向你展示了高斯分布的概率密度函数。
   每个样本都有一个取决于其特征取值的位置 (x1, x2)。"""
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # 设置标题
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # 设置 y 轴标签
    plt.ylabel('Throughput (mb/s)')
    # 设置 x 轴标签
    plt.xlabel('Latency (ms)')