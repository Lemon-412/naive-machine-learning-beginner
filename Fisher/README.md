# Fisher 线性分类器

**Fisher线性分类器的判别思想**：

在超平面上选择一个投影向量 $\boldsymbol{w^*}$ ，在各类别投影后各类相隔尽可能远，而类内相隔尽可能近。



**定义**：

训练样本集 $X=\left\{ x_1, x_2, \cdots, x_N \right\}$ 

其中 $w_i$ 类样本集 $X_i = \left\{x_1^i,x_2^i, \cdots, x_{N_i}^i \right\}$

各类均值向量 $\boldsymbol{m}_i = \frac {1} {N_i} \sum_{\boldsymbol{x}_j \in X_i} \boldsymbol{x}_j$

各类离散度矩阵 $\boldsymbol{S}_i = \sum_{\boldsymbol{x}_j \in X_i} (\boldsymbol{x}_j - \boldsymbol{m}_i) (\boldsymbol{x}_j - \boldsymbol{m}_i)^\top	$ ， $\boldsymbol{S}_w = \boldsymbol{S}_0 + \boldsymbol{S}_1$

则有： $\boldsymbol{w^*}= \boldsymbol{S}_w^{-1}(\boldsymbol{m}_0-\boldsymbol{m}_1)$ 



**证明**：

[WIP]

