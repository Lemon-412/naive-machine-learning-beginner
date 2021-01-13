# Fisher 线性分类器

**Fisher线性分类器(LDA)的判别思想**：

在超空间中选择一个投影向量 $\boldsymbol{w^*}$ ，在各类别投影后各类相隔尽可能远，而类内相隔尽可能近。

设定阈值 $w_0$ ，则以 $\boldsymbol{w}^*$ 为法向量， $w_0$ 为截距的超平面即决策面。

定义准则函数 $J_F(w) = \frac{S_b}{S_w} = \frac{\left( m_1-m_2 \right)^2}{S_1^2+S_2^2}$

则 $\boldsymbol{w}^* = \arg\max_{\boldsymbol{w}}J_F(\boldsymbol{w})$



**定义**：

训练样本集 $X=\left\{ x_1, x_2, \cdots, x_N \right\}$ 

其中 $\omega_i$ 类样本集 $X_i = \left\{x_1^i,x_2^i, \cdots, x_{N_i}^i \right\}$

各类均值向量 $\boldsymbol{m}_i = \frac {1} {N_i} \sum_{\boldsymbol{x}_j \in X_i} \boldsymbol{x}_j$

各类离散度矩阵 $\boldsymbol{S}_i = \sum_{\boldsymbol{x}_j \in X_i} (\boldsymbol{x}_j - \boldsymbol{m}_i) (\boldsymbol{x}_j - \boldsymbol{m}_i)^\top	$ ， $\boldsymbol{S}_w = \boldsymbol{S}_0 + \boldsymbol{S}_1$

则有： $\boldsymbol{w^*}= \boldsymbol{S}_w^{-1}(\boldsymbol{m}_0-\boldsymbol{m}_1)$ 



**证明**：

[WIP]



**预测**：

决策面即 $\boldsymbol{w}^\top \boldsymbol{x} + w_0 = 0 $

代入计算 $\hat y = \boldsymbol{w}^\top \boldsymbol{x} + w_0$ 若 $\hat y > 0$ 则分为正类，否则分为反类。

其中，当样本正态分布且两类协方差相同时，分类阈值可以定义为：

$\omega_0 = - \frac{1}{2} \left( \boldsymbol{m}_1 + \boldsymbol{m}_2 \right)^\top \boldsymbol{S}_w^{-1} \left( \boldsymbol{m}_1 - \boldsymbol{m}_2 \right) - \frac{ \ln \left| w_2 \right|  - \ln \left| w_1 \right| } {\left| w_1 \right| + \left| w_2 \right| + 2} $