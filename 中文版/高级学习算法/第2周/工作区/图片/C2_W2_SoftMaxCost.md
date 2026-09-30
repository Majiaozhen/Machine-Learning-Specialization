# 图片文字中文翻译

> 原图：C2_W2_SoftMaxCost.png

代价：Softmax 回归 vs 逻辑回归
逻辑回归：loss = −y·log(a1) − (1 − y)·log(1 − a1)
Softmax 回归：a_j = e^z_j/(e^z1 + e^z2 + … + e^zN)
交叉熵损失 = −log a2 若 y = 2；…；−log aN 若 y = N
J(w,b) = 损失的平均值
