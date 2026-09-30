import numpy as np

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
    # s = 
    ### 在此处编写代码 ###（约 1 行代码）
    s = 1/(1 + np.exp(-z))
    ### 代码编写结束 ###
    
    return s


def compute_cost(X, y, w, b): 
    m = X.shape[0]

    f_w = sigmoid(np.dot(X, w) + b)
    total_cost = (1/m)*np.sum(-y*np.log(f_w) - (1-y)*np.log(1-f_w))
    
    return float(np.squeeze(total_cost))

def compute_gradient(X, y, w, b):
    """计算逻辑回归的梯度。
   
    参数
    ----------
    X : array_like
        形状 (m, n+1)
   
    y : array_like
        形状 (m,)
   
    w : array_like
        模型参数
        形状 (n+1,)
    b:  scalar
   
    返回
    -------
    dw : array_like
        形状 (n+1,)
        梯度
    db: scalar"""
    m = X.shape[0]
    f_w = sigmoid(np.dot(X, w) + b)
    err = (f_w - y)
    dw = (1/m)*np.dot(X.T, err)
    db = (1/m)*np.sum(err)
    
    return float(np.squeeze(db)), dw


def predict(X, w, b): 
    """使用学习到的逻辑回归参数 theta 预测标签是 0 还是 1
   
    参数
    ----------
    X : array_like
        形状 (m, n+1)
   
    w : array_like
        模型参数
        形状 (n, 1)
    b : scalar
   
    返回
    -------

    p: array_like
        形状 (m,)
        以 0.5 为阈值对 X 做出的预测
        即若 sigmoid (theta.T*X) >=0.5 则预测为 1"""
    
    # 训练样本数
    m = X.shape[0]   
    p = np.zeros(m)
   
    for i in range(m):
        f_w = sigmoid(np.dot(w.T, X[i]) + b)
        p[i] = f_w >=0.5
    
    return p

def compute_cost_reg(X, y, w, b, lambda_=1): 
    """计算带正则化的逻辑回归代价
   
    参数
    ----------
    X : array_like
        形状 (m, n+1)
   
    y : array_like
        形状 (m,)
   
    w: array_like
        模型参数
        形状 (n+1,)
    b: scalar
   
    返回
    -------
    cost : float
        以 theta 为逻辑回归参数来拟合 X 和 y 中数据点的代价"""
    # 训练样本数
    m = X.shape[0]
    
    # 你需要正确返回以下变量
    cost = 0

    f = sigmoid(np.dot(X, w) + b)
    reg = (lambda_/(2*m)) * np.sum(np.square(w))
    cost = (1/m)*np.sum(-y*np.log(f) - (1-y)*np.log(1-f)) + reg
    return cost


def compute_gradient_reg(X, y, w, b, lambda_=1): 
    """计算带正则化的逻辑回归梯度
   
    参数
    ----------
    X : array_like
        形状 (m, n+1)
   
    y : array_like
        形状 (m,)
   
    w : array_like
        模型参数
        形状 (n+1,)
    b : scalar
   
    返回
    -------
    db: scalar
    dw: array_like
        形状 (n+1,)"""
    # 训练样本数
    m = X.shape[0]
    
    # 你需要正确返回以下变量
    cost = 0
    dw = np.zeros_like(w)

    f = sigmoid(np.dot(X, w) + b)
    err = (f - y)
    dw = (1/m)*np.dot(X.T, err)
    dw += (lambda_/m)  * w
    db = (1/m) * np.sum(err)
 
    #print(db,dw)

    return db,dw
