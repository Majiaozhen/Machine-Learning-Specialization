# 图片文字中文翻译

> 原图：C2_W2_Softmax_Header.PNG

逻辑回归
Softmax 回归（4 个可能的输出）：
a1 = e^z1/(e^z1 + e^z2 + e^z3 + e^z4) = P(y = 1|x) = 0.30
a2 = … = P(y = 2|x) = 0.20
a3 = … = P(y = 3|x) = 0.15
a4 = … = P(y = 4|x) = 0.35
带 Softmax 输出的神经网络
输入 → 25 个单元 → 15 个单元 → 10 个单元
最后一层做逻辑回归，输出经 softmax
Softmax 回归（N 个可能的输出）：a_j = P(y = j|x)
