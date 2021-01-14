# PCA 变换

输入样本为$D = \{x_1, x_2, \dots, x_m\}$

对所有样本进行$min-max $ $Normalization$

对样本进行均值化$x_i \leftarrow x_i - \frac 1 m \sum_{i=1}^m x_i$

计算样本的协方差矩阵$X X^T$

取最大的$d$ 个特征值和特征向量$w_1, w_2, \dots w_d$

投影矩阵即为$W = (w_1, w_2, \dots w_d)$

用$W$对样本进行投影变换至低维