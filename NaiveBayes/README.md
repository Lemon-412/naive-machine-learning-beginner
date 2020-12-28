# Naïve Bayes

### 贝叶斯定理

设$C$为标签集，$\boldsymbol X$为特征集，$c\in C$， $\boldsymbol x \in \boldsymbol X$

贝叶斯定理：$ P \left( c| \boldsymbol x \right) = \frac{P\left(c\right)P\left(\boldsymbol x|c\right)}{p\left(x\right)}$



### 朴素贝叶斯分类器

设各个特征之间全部相互独立， 则有$P\left(\boldsymbol x|c\right) = \prod_{i}P\left(\boldsymbol x_i|c\right)$

朴素贝叶斯分类器：若将最小错误率作为学习目标，则有

$\hat c = \arg\max_{c}P \left( c| \boldsymbol x \right)=\arg\max_c P\left(c\right)\prod_{i}P\left(\boldsymbol x_i|c\right)$

其中$P \left( c\right)=\frac{\left| D_c \right|}{\left|D\right|}$ ，$P\left(\boldsymbol x_i|c\right)$对离散型特征和连续型特征有不同的处理方法。



### 离散型特征

设$D$为测试集，$D_c$为测试集中所有标签为$c$的数据集合，$D_{c,\boldsymbol x_i}$为测试集中所有标签为$c$且第$i$个特征值为$\boldsymbol {x}_i$的数据集合。

$P\left(\boldsymbol x_i|c\right) = \frac{\left|D_{c,\boldsymbol x_i}\right|}{\left| D_c \right|}$



### 连续型特征

设特征服 $\boldsymbol x$ 从正态分布，通过数据集估计出再各个 $c$ 中 $\boldsymbol x$的分布特征$\mu$和$\sigma ^2$

$P\left(\boldsymbol x_i|c\right) = \frac {1} { \sqrt{ 2\pi \sigma_{c,i}^2 } } \exp \left( - \frac {\left( \boldsymbol x_i - \mu_{c,i} \right) ^2 } {2 \sigma_{c,i} ^2 } \right)$



### 对连乘下溢进行优化

对等式进行变化，避免连乘造成浮点数下溢。

$\hat c = \arg\max_c \log \left(P\left(c\right)\prod_{i}P\left(\boldsymbol x_i|c\right) \right) = \arg\max_c \log P\left(c\right) + \sum_i \log P\left(\boldsymbol x_i|c \right)$



### 离散特征的修正

对于离散特征，为了避免由于$\boldsymbol {x}_i$对应$c$在数据集中没有出现，而被贝叶斯分类器忽视其他特征的一票否决，即$P\left(\boldsymbol x_i|c\right)=0$直接导致$P\left(c|\boldsymbol x\right) = 0$

设$N$为分类数，$N_i$为第$i$个特征分类数。可以进行如下修正：

$P \left( c\right)= \frac{ \left| D_c \right| + 1}{\left|D\right| + N}$ ，$P\left(\boldsymbol x_i|c\right) = \frac{\left|D_{c,\boldsymbol x_i}\right| + 1}{\left| D_c \right| + N_i}$

随着数据集大小增长，修正对结果的影响将趋于0。



### 最小风险决策

对于二分类$C = \left\{c_1 , c_2 \right\}$问题，定义损失函数表$\lambda_{i,j}$为将$i$预测为$j$的风险（损失）

定义$l\left( \boldsymbol x \right) = \frac {P\left( \boldsymbol x|c_1 \right)} {P\left( \boldsymbol x|c_2 \right)}$，

若$l\left( \boldsymbol x \right) > \frac {\left( \lambda_{12} -\lambda_{22} \right) P\left( c_2 \right)} {\left( \lambda_{21} -\lambda_{11} \right) P\left( c_1 \right)}$，则$\hat c = c_1$

若$l\left( \boldsymbol x \right) < \frac {\left( \lambda_{12} -\lambda_{22} \right) P\left( c_2 \right)} {\left( \lambda_{21} -\lambda_{11} \right) P\left( c_1 \right)}$，则$\hat c = c_2$



### ROC

对于二分类问题，将上述等式进行变换，有$\frac {P\left( \boldsymbol x|c_1 \right) P\left( c_1 \right)} {P\left( \boldsymbol x|c_2 \right) P\left( c_2 \right)} > f \left( \lambda \right)$

考虑用$ f\left( \lambda \right) $作为分类阈值，则对每一个样本$ \boldsymbol x$，$g\left( \boldsymbol x\right) = \frac {P\left( \boldsymbol x|c_1 \right) P\left( c_1 \right)} {P\left( \boldsymbol x|c_2 \right) P\left( c_2 \right)}$为常数。

根据$g\left( \boldsymbol x \right)$排序可以绘制ROC曲线。

