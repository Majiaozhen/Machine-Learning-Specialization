# lab_utils_common
#   包含常用程序和变量定义
#   供本周所有实验使用。
#   相比之下，专用的、大型的绘图程序会放在单独的文件中
#   一般会导入到使用它们的当周。
#   那些文件会导入本文件

import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import interact, Output

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

plt.style.use('./deeplearning.mplstyle')
plt.rcParams['font.size'] = 8

import copy
import math


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
    z = np.clip( z, -500, 500 )           # 防止溢出
    g = 1.0/(1.0+np.exp(-z))
    
    return g

##########################################################
# 回归相关程序
##########################################################
       
def predict_logistic(X, w, b):
    return(sigmoid(X @ w + b))

def predict_linear(X, w, b):
    return(X @ w + b)
    
def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """用逻辑损失计算代价，非矩阵版本

    参数：
      X (ndarray): 形状 (m,n)  含 n 个特征的样本矩阵
      y (ndarray): 形状 (m,)   目标值
      w (ndarray): 形状 (n,)   用于预测的参数
      b (scalar):               用于预测的参数
      lambda_ : (scalar, float) 控制正则化强度，0 = 不正则化
    返回：
      cost (scalar): 代价"""

    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i    = np.dot(X[i],w) + b                                             #(n,)(n,) or (n,) ()
        if safe:  #避免溢出
            cost += -(y[i] * z_i ) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)                                                   #(n,)
            cost  += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)       # 标量
    cost = cost/m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)                                               # 标量
        reg_cost = (lambda_/(2*m))*reg_cost
    
    return cost + reg_cost

def compute_cost_linear_matrix(X, y, w, b, verbose=False):
    """用平方误差损失、并借助矩阵运算提速来计算代价
     参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
      verbose : (Boolean) 若为真，打印中间值 f_wb
    返回
      total_cost: (scalar)                 代价""" 
    m,n = X.shape

    # 计算所有样本的 f_wb。
    f_wb = X @ w + b  
    # 计算代价
    total_cost = (1/(2*m)) * np.sum((f_wb-y)**2)

    if verbose: print(f"f_wb: {f_wb}")       
    return total_cost

def compute_gradient_linear_matrix(X, y, w, b): 
    """借助矩阵运算提速，计算线性回归平方误差代价的梯度
 
    参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
    返回
      dj_dw: (array_like 形状 (n,1))     代价对参数 w 的梯度。
      dj_db: (scalar)                     代价对参数 b 的梯度。"""
    m,n    = X.shape
    y = y.reshape(-1,1)             # 确保是二维
    w = w.reshape(-1,1)             # 确保是二维
    f_wb   = X @ w + b              
    err      = f_wb - y                
    dj_dw  = (1/m) * (X.T @ err)    
    dj_db  = (1/m) * np.sum(err)    
        
    return dj_db,dj_dw


def compute_cost_logistic_matrix(X, y, w, b, verbose=False):
    """借助矩阵运算提速，用逻辑损失计算代价
    参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
      verbose : (Boolean) 若为真，打印中间值 f_wb
    返回：
      total_cost: (scalar)                代价"""
    m = X.shape[0]
    y = y.reshape(-1,1)             # 确保是二维
    w = w.reshape(-1,1)             # 确保是二维
    f = sigmoid(X @ w + b)          #(m,n)(n,1) = (m,1)
    total_cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   #(1,m)(m,1)=(1,1)
    
    if verbose: print(f"f: {f}")       
    return total_cost[0,0]


def compute_gradient_logistic_matrix(X, y, w, b): 
    """用矩阵计算逻辑回归代价的梯度
 
    参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
    返回
      dj_dw: (array_like 形状 (n,1))     代价对参数 w 的梯度。
      dj_db: (scalar)                     代价对参数 b 的梯度。"""
    m,n   = X.shape
    y = y.reshape(-1,1)             # 确保是二维
    w = w.reshape(-1,1)             # 确保是二维
    f_wb  = sigmoid(X @ w + b)      
    err   = f_wb - y                 
    dj_dw = (1/m) * (X.T @ err)      
    dj_db = (1/m) * np.sum(err)     

    return dj_db,dj_dw

def log_1pexp(x, max=20):
    '''近似计算 log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    参数：
    x   : (ndarray 形状 (n,1) 或 (n,)  输入
    out : (ndarray 形状与 x 相同      输出 ~= np.log(1+exp(x))'''

    out  = np.zeros_like(x,dtype=float)
    i    = x <= max
    ni   = np.logical_not(i)
    
    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return(out)


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    """用矩阵计算代价
    参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
      verbose : (Boolean) 若为真，打印中间值 f_wb
    返回：
      total_cost: (scalar)                代价"""
    m = X.shape[0]
    y = y.reshape(-1,1)             # 确保是二维
    w = w.reshape(-1,1)             # 确保是二维
    if logistic:
        if safe:  #免于溢出
            z = X @ w + b
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m                                                   # （标量）
        else:
            f    = sigmoid(X @ w + b)                                               # (m,n)(n,1) = (m,1)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            cost = cost[0,0]                                                        # 标量
    else: 
        f    = X @ w + b                                                        # (m,n)(n,1) = (m,1)
        cost = (1/(2*m)) * np.sum((f - y)**2)                                   # 标量

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                   # 标量

    total_cost = cost + reg_cost                                                # 标量

    return total_cost                                                           # 标量

def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0): 
    """用矩阵计算梯度
 
    参数：
      X : (ndarray, 形状 (m,n))          样本矩阵
      y : (ndarray  形状 (m,) 或 (m,1))  每个样本的目标值
      w : (ndarray  形状 (n,) 或 (n,1))  模型参数取值
      b : (scalar )                       模型参数取值
      logistic: (boolean)                 为假时用线性，为真时用逻辑
      lambda_:  (float)                   非零时施加正则化
    返回
      dj_dw: (array_like 形状 (n,1))     代价对参数 w 的梯度。
      dj_db: (scalar)                     代价对参数 b 的梯度。"""
    m,n = X.shape
    y = y.reshape(-1,1)             # 确保是二维
    w = w.reshape(-1,1)             # 确保是二维
    
    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b      # (m,n)(n,1) = (m,1)
    err   = f_wb - y                                              # (m,1)
    dj_dw = (1/m) * (X.T @ err)                                   # (n,m)(m,1) = (n,1)
    dj_db = (1/m) * np.sum(err)                                   # 标量
    
    dj_dw += (lambda_/m) * w        # 正则化                  # (n,1)

    return dj_db, dj_dw                                           # 标量, (n,1)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True): 
    """执行批量梯度下降以学习 theta。以学习率 alpha 走
    num_iters 个梯度步来更新 theta
   
    参数：
      X (ndarray):    形状 (m,n)         样本矩阵
      y (ndarray):    形状 (m,) 或 (m,1) 每个样本的目标值
      w_in (ndarray): 形状 (n,) 或 (n,1) 模型参数的初始值
      b_in (scalar):                      模型参数的初始值
      logistic: (boolean)                 为假时用线性，为真时用逻辑
      lambda_:  (float)                   非零时施加正则化
      alpha (float):                      学习率
      num_iters (int):                    运行梯度下降的迭代次数
     
    返回：
      w (ndarray): 形状 (n,) 或 (n,1)    更新后的参数值；与传入形状一致
      b (scalar):                         更新后的参数值
          运行梯度下降"""
    # 训练样本数
    m = len(X)
    
    # 用于存放代价 J 以及每次迭代的 w，主要供后续绘图使用
    J_history = []
    w = copy.deepcopy(w_in)  # 避免在函数内修改全局变量 w
    b = b_in
    w = w.reshape(-1,1)      #为矩阵运算做准备
    y = y.reshape(-1,1)

    for i in range(num_iters):

        # 计算梯度并更新参数
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)   

        # 用 w、b、alpha 和梯度更新参数
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # 每次迭代保存代价 J
        if i<100000:      # 防止资源耗尽
            J_history.append( compute_cost_matrix(X, y, w, b, logistic, lambda_) )

        # 每 10 次迭代打印一次代价；若迭代次数少于 10，则打印所有迭代
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w.reshape(w_in.shape), b, J_history, #返回最终的 w、b 以及用于绘图的 J 历史

def oldgradient_descent(X, y, w_in, b_in, alpha, num_iters,cost_function, gradient_function, verbose=True): 
    """执行批量梯度下降以学习 theta。以学习率 alpha 走
    num_iters 个梯度步来更新 theta
   
    参数：
      X (ndarray):    形状 (m,n) 样本矩阵
      y (ndarray):    形状 (m,)  每个样本的目标值
      w_in (ndarray): 形状 (n,)  模型参数的初始值
      b_in (scalar):              模型参数的初始值
      cost_function:     用于计算代价的函数
      gradient_function: 用于计算梯度的函数
      alpha (float):     学习率
      num_iters (int): 运行梯度下降的迭代次数
     
    返回：
      w (ndarray): 形状 (n,) 运行梯度下降后更新得到的模型参数值
      b (scalar):            运行梯度下降后更新得到的模型参数值"""
    # 训练样本数
    m = len(X)
    
    # 用于存放代价 J 以及每次迭代的 w，主要供后续绘图使用
    J_history = []
    w = copy.deepcopy(w_in)  # 避免在函数内修改全局变量 w
    b = b_in
    for i in range(num_iters):

        # 计算梯度并更新参数
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # 用 w、b、alpha 和梯度更新参数
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # 每次迭代保存代价 J
        if i<100000:      # 防止资源耗尽
            J_history.append( cost_function(X, y, w, b))

        # 每 10 次迭代打印一次代价；若迭代次数少于 10，则打印所有迭代
        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history, #返回最终的 w、b 以及用于绘图的 J 历史

def zscore_normalize_features(X):
    """计算按列做 z 分数归一化后的 X
   
    参数：
      X (ndarray): 形状 (m,n)，输入数据，m 个样本，n 个特征
     
    返回：
      X_norm (ndarray): 形状 (m,n)，按列归一化后的输入
      mu (ndarray):     形状 (n,)，每个特征的均值
      sigma (ndarray):  形状 (n,)，每个特征的标准差"""
    # 求每一列/每个特征的均值
    mu     = np.mean(X, axis=0)                 # mu 的形状为 (n,)
    # 求每一列/每个特征的标准差
    sigma  = np.std(X, axis=0)                  # sigma 的形状为 (n,)
    # 逐元素操作：从每个样本的该列中减去该列的均值，再除以该列的标准差
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#检查一下我们的结果
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

######################################################
# 常用绘图程序
######################################################


def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    
    # 找出正例与负例的索引
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #可处理一维或一维 y 向量
    neg = neg.reshape(-1,)

    # 绘制样本
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)
    
    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def plt_tumor_data(x, y, ax):
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="malignant")
    ax.scatter(x[neg], y[neg], marker='o', s=100, label="benign", facecolors='none', edgecolors=dlblue,lw=3)
    ax.set_ylim(-0.175,1.1)
    ax.set_ylabel('y')
    ax.set_xlabel('Tumor Size')
    ax.set_title("Logistic Regression on Categorical Data")
    
    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

    

# 在 0.5 处画一条阈值线
def draw_vthresh(ax,x):
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
        ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
        ax.annotate("z >= 0", xy= [x,0.5], xycoords='data', 
             xytext=[30,5],textcoords='offset points')
        d = FancyArrowPatch(
            posA=(x, 0.5), posB=(x+3, 0.5), color=dldarkred,
            arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
        )
        ax.add_artist(d)

        ax.annotate("z < 0", xy= [x,0.5], xycoords='data', 
                     xytext=[-50,5],textcoords='offset points', ha='left')
        f = FancyArrowPatch(
            posA=(x, 0.5), posB=(x-3, 0.5), color=dlblue,
            arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
        )
        ax.add_artist(f)
        
