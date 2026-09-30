# 图片文字中文翻译

> 原图：C2_W2_softmax_accurate.png

Softmax 的数值上更精确的实现
Softmax 回归：loss = −log a1 若 y = 1；…；loss = −log a10 若 y = 10
loss = −log(e^z1 + … + e^z10) 若 y = 10（更精确）
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
