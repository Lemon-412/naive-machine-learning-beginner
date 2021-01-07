# Reinforcement Learning

### Scenario of RL

RL由**Agent**和**Environment**组成：

- **State**(Observation)：Agent 观测 Environment。DRL的强表达能力使得可以直接将Observation作为State，而不需要做额外的Summary。
- **Action**： Agent 产生行为改变 Environment，Environment 给予 **reward**

- Agent learns to take actions to maximize expected reward.



### Learning from experience

RL从环境学习的过程中产生一系列的Action，但直到到达终态（完成一个**episode** $\tau$），大多数的reward可能都是0，缺少对每一个行为的好坏的评判。因此，相较于supervised learning，RL更需要从经验中进行学习（reward delay）。



### 马尔科夫决策过程对RL的描述

定义环境 $E=<X,A,P,R>$

- 状态空间$X$：对Agent感知到的环境的描述。$\forall x \in X$唯一对应一个环境的状态（State）。

- 动作空间$A$：对Agent能采取的行为的约束。$\forall a \in A$唯一对应一个可采取的行为。在一些问题中，$a|x$。

- 转移函数$P$：$X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作下转移到另一状态的概率。黑盒。

- 奖赏$R$：$X \times X \rightarrow \mathbb{R}$ 或 $X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作转移到某一状态的奖赏（reward）。黑盒。



### policy based (Actor $\pi$ )

定义策略$\pi$：$X \times A \rightarrow \mathbb{R}$。即 $\pi\left( x,a \right)$ 表示状态 $x$ 下执行动作 $a$ 的概率。这是 Actor 的学习目标。

**Actor**： $\pi _{\theta} \left( x,a \right)$:

- input： State，一般用一个向量或矩阵表示
- output： Action，采取每一个action对应的几率



定义**Trajectory**：$\tau = \left\{ s_1,a_1,r_1,s_2,a_2,r_2,\cdots , s_T,a_T,r_T \right\}$

定义**Total Reward**：$R\left( \tau \right) = \sum_{n=1}^{N} r_n$

一个episode的期望reward：$\bar{R}_\theta = \mathbb {E} \left| R_{\theta} \right| = \sum_{\tau} R\left( \tau \right) P\left( \tau | \theta \right)$

使用 $\pi_{\theta}$ 进行N次实验，则有：$\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} R\left(  \tau^n \right)$



最优的参数即使得游戏期望累计收益最大的参数。

$\theta ^* = \arg\max \bar{R}_\theta$：可以使用梯度上升gradient ascent。

$\theta^{new} = \theta^{old} + \eta \nabla \bar{R}_{\theta^{old}}$ ，其中 $\eta$ 为学习率learning rate。



对 $\theta$ 求梯度：

$\nabla \bar{R}_\theta = \sum_{\tau} R(\tau) \nabla P(\tau|\theta) = \sum_{\tau} R(\tau) P(\tau|\theta) \frac {\nabla P(\tau|\theta)} {P(\tau|\theta)} = \sum_{\tau} R(\tau) P(\tau|\theta) \nabla \log P(\tau|\theta)$

其中 $R(\tau)$ 与 $\theta$ 无关，无需求导，由 $E$ 给出（黑盒）

由于：$ \sum_{\tau}P(\tau|\theta)f(\cdot) \approx \frac{1}{N}f(\cdot)$

使用 $\pi_\theta$ 进行N局游戏，获得 $\left\{ \tau^1, \tau^2, \cdots \tau^n \right\}$，则有：

$\nabla \bar{R}_\theta \approx \frac{1}{N}\sum_{n=1}^N R \left( \tau^n \right) \nabla \log P \left( \tau^n | \theta \right)$



求 $\nabla \log P \left( \tau^n | \theta \right)$：

$ P(\tau|\theta) = p(s_1) \prod_{t=1}^T p(a_t|s_t, \theta) p(r_t, s_{t+1} | s_t, a_t)$，其中 $p(s_1)$ 和 $p(r_t, s_{t+1} | s_t, a_t)$ 由 $E$ 给出

$\nabla \log P(\tau|\theta) = \sum_{t=1}^T \nabla \log (a_t|s_t, \theta)$



所以：

$\nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} R(\tau^n) \nabla \log(a_t^n|s_t^n, \theta)$

即进行N局游戏后，每局游戏每一时刻采取策略的概率与该局游戏总收益的乘积的和的梯度。

注意：收益使用总收益而非单步收益。



为什么刻意使用  $\nabla \log P(\tau|\theta)$ 代替 $\nabla P(\tau|\theta)$？

- 将 $\sum_{\tau}P(\tau|\theta)f(\cdot )$ 转换为 $ \frac{1}{N}f(\cdot)$求解
- 将 $P(\tau|\theta)$ 中的连乘转换为连加求解
- 直观上除以 $P(\tau|\theta )$ 使得 $\pi_\theta$ 中不常规的却高收益的行为 $\tau_t$ 有了被大幅更新的可能



(trick)添加Baseline：

由于actor对各个行为的概率会做归一化处理，即 $\sum_{a} \pi_\theta(a|s) = 1$ ，当 $R(\cdot)$恒为正或恒为负时，理论上不会影响学习。

但，如果 $R(\cdot)$ 恒正，可能会陷入一些一开始没被sample到的action，在更新梯度后更不易被sample到的窘境。

可以考虑添加baseline，设计参数 $b$ 并修正为上式为 $\nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} R(\tau^n - b) \nabla \log(a_t^n|s_t^n, \theta)$



### Value Based (Critic)

