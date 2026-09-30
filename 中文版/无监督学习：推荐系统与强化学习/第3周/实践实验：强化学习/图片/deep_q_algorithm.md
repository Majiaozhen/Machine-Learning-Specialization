# 图片文字中文翻译

> 原图：deep_q_algorithm.png

算法 1：带经验回放的深度 Q 学习（Deep Q-Learning with Experience Replay）
1 初始化容量为 N 的记忆缓冲区 D
2 用随机权重初始化 Q 网络
3 用权重……初始化目标 Q 网络
4 for episode = 1 to … do
    接收初始观测状态 S1
5   for t = 1 to … do
6     观测状态 St，并用 ε-贪心策略选择动作 At
7     在环境中执行动作 At，获得奖励 Rt 和下一状态 St+1
8     把经验元组 (St, At, Rt, St+1) 存入记忆缓冲区 D
9     每隔 C 步执行一次学习更新：
10        从 D 中随机抽取一个小批量的经验元组 (Sj, Aj, Rj, Sj+1)
11        若该回合在第 j+1 步终止，则令 yj = Rj；否则令 yj = Rj + γ·max_a' Q(sj+1, a')
12        针对 Q 网络权重，对 (yj − Q(sj, aj))² 执行一步梯度下降
13        用软更新（soft update）更新 Q 网络的权重
    end
end
