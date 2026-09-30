import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("数据/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # 绘制 X 中的数据点并着色，使具有相同
    # idx 中相同索引分配的样本颜色相同
    plt.scatter(X[:, 0], X[:, 1], c=idx)
    
def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # 绘制样本
    plot_data_points(X, idx)
    
    # 用黑色 'x' 标出质心
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # 用折线绘制质心的历史
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)