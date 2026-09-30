# 图片文字中文翻译

> 原图：C2_W2_Softmax.png

逻辑回归：z = w·x + b，g(z) = 1/(1 + e^(−z)) = P(y = 1|x)
Softmax 回归（N 个可能的输出）：
z_j = w_j·x + b_j，a_j = e^z_j/(e^z1 + … + e^zN) = P(y = j|x)
Softmax 回归（4 个可能的输出）：
a1 = e^z1/(e^z1 + e^z2 + e^z3 + e^z4) = P(y = 1|x) = 0.30
a2 = e^z2/(…) = P(y = 2|x) = 0.20
a3 = e^z3/(…) = P(y = 3|x) = 0.15
a4 = e^z4/(…) = P(y = 4|x) = 0.35
