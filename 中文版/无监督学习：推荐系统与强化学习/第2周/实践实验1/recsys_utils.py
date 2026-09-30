import numpy as np
import pandas as pd
from numpy import loadtxt

def normalizeRatings(Y, R):
    """预处理数据：对每部电影（每一行）减去平均评分。
    只考虑真实评分 R(i,j)=1。
    [Ynorm, Ymean] = normalizeRatings(Y, R) 对 Y 做归一化，使每部电影
    的平均评分为 0。未评分的电影此时平均评分为 (0)。
    在 Ymean 中返回平均评分。"""
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():

    file = open('./数据/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./数据/small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./数据/small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./数据/small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./数据/small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """返回一个 df，其索引为电影在 Y 矩阵中的排列顺序"""
    df = pd.read_csv('./数据/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)





