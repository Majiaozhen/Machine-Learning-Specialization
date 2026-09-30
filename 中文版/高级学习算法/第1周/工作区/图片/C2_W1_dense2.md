# 图片文字中文翻译

> 原图：C2_W1_dense2.PNG

神经网络层
def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:, j], a_in) + b[j]
        a_out[j] = g(z)
    return a_out
