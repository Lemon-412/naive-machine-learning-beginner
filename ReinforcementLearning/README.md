# Reinforcement Learning



> 参考资料：李宏毅强化学习2020
>
> https://www.bilibili.com/video/BV1UE411G78S?p=4
>
> 已完成：p1, p2, p3, p4, p5



### Scenario of RL

RL由**Agent**和**Environment**组成：

- **State**(Observation)：Agent 观测 Environment。DRL的强表达能力使得可以直接将Observation作为State，而不需要做额外的Summary。
- **Action**： Agent 产生行为改变 Environment，Environment 给予 **reward**

- Agent learns to take actions to maximize expected reward.



### Learning from experience

RL从环境学习的过程中产生一系列的Action，但直到到达终态（完成一个**episode** $\tau$），大多数的reward可能都是0，缺少对每一个行为的好坏的评判。因此，相较于supervised learning，RL更需要从经验中进行学习（reward delay）。



> Inversed Reinforcement Learning (IRL)
>
> 通常RL是给定environment，通过黑盒的reward function去学一个 $\theta$ 
>
> IRL用于解决现实中reward function复杂甚至无法定义的情况，给定environment，并通过expert的知道去学习一个reward function。
>
> > 如何定义无人驾驶的reward function？撞了人扣一百分，那撞了狗呢？
>
> 具体实现方法类似GAN，Actor始终模仿expert的行为（最终成为generator），而Discriminator始终学着区分Actor和expert（最终成为reward function）



### 马尔科夫决策过程对RL的描述

定义环境 $E=<X,A,P,R>$

- 状态空间$X$：对Agent感知到的环境的描述。$\forall x \in X$唯一对应一个环境的状态（State）。

- 动作空间$A$：对Agent能采取的行为的约束。$\forall a \in A$唯一对应一个可采取的行为。在一些问题中，$a|x$。

- 转移函数$P$：$X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作下转移到另一状态的概率。黑盒。

- 奖赏$R$：$X \times X \rightarrow \mathbb{R}$ 或 $X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作转移到某一状态的奖赏（reward）。黑盒。



### Policy Gradient

定义策略 $\pi$ ：$X \times A \rightarrow \mathbb{R}$。即 $\pi\left( x,a \right)$ 表示状态 $x$ 下执行动作 $a$ 的概率。这是 Actor 的学习目标。

**Actor**(policy)： $\pi _{\theta} \left( x,a \right)$:

- input： State，一般用一个向量或矩阵表示
- output： Action，采取每一个action对应的几率



定义**Trajectory**：$\tau = \left\{ s_1,a_1,r_1,s_2,a_2,r_2,\cdots , s_T,a_T,r_T \right\}$

定义**Total Reward**：$R\left( \tau \right) = \sum_{n=1}^{N} r_n$

定义 $\theta$ 参数下，期望的 total reward：$\bar{R}_\theta = \mathbb {E} \left| R_{\theta} \right| = \sum_{\tau} R\left( \tau \right) P\left( \tau | \theta \right)$

如果使用 $\pi_{\theta}$ 进行N次实验，则有：$\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} R\left(  \tau^n \right)$



最优的参数即使得游戏期望累计收益最大的参数。

$\theta ^* = \arg\max \bar{R}_\theta$：可以使用梯度上升 **Gradient Ascent**。

$\theta^{new} = \theta^{old} + \eta \nabla \bar{R}_{\theta^{old}}$ ，其中 $\eta$ 为学习率 learning rate。



对 $\theta$ 求梯度：

$\displaystyle \nabla \bar{R}_\theta = \sum_{\tau} R(\tau) \nabla P(\tau|\theta) = \sum_{\tau} R(\tau) P(\tau|\theta) \frac {\nabla P(\tau|\theta)} {P(\tau|\theta)} = \sum_{\tau} R(\tau) P(\tau|\theta) \nabla \log P(\tau|\theta)$

其中 $R(\tau)$ 与 $\theta$ 无关，无需求导，可由 $E$ 直接给出（黑盒）

由于：$ \sum_{\tau}P(\tau|\theta)f(\cdot) \approx \frac{1}{N}f(\cdot)$

如果使用 $\pi_\theta$ 进行N局游戏，获得 $\left\{ \tau^1, \tau^2, \cdots \tau^n \right\}$，则有：

$\displaystyle \nabla \bar{R}_\theta \approx \frac{1}{N}\sum_{n=1}^N R \left( \tau^n \right) \nabla \log P \left( \tau^n | \theta \right)$



求 $\nabla \log P \left( \tau^n | \theta \right)$ (即 $\nabla p_\theta(\tau)$)：

$ P(\tau|\theta) = p(s_1) \prod_{t=1}^T p(a_t|s_t, \theta) p(r_t, s_{t+1} | s_t, a_t)$，其中 $p(s_1)$ 和 $p(r_t, s_{t+1} | s_t, a_t)$ 由 $E$ 给出

$\displaystyle \nabla \log P(\tau|\theta) = \sum_{t=1}^T \nabla \log (a_t|s_t, \theta)$



所以：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} R(\tau^n) \nabla \log(a_t^n|s_t^n, \theta)$

即进行N局游戏后，每局游戏每一时刻采取策略的概率与该局游戏总收益的乘积的和的梯度。

注意：收益使用 $\tau^n$ 的总收益而非单步收益。



为什么刻意使用  $\nabla \log P(\tau|\theta)$ 代替 $\nabla P(\tau|\theta)$？

- 将 $\sum_{\tau}P(\tau|\theta)f(\cdot )$ 转换为 $ \frac{1}{N}f(\cdot)$求解
- 将 $P(\tau|\theta)$ 中的连乘转换为连加求解
- 直观上除以 $P(\tau|\theta )$ 使得 $\pi_\theta$ 中不常规的却高收益的行为 $\tau_t$ 有了被大幅更新的可能



**添加Baseline** (trick)：

由于actor对各个行为的概率会做归一化处理，即 $\sum_{a} \pi_\theta(a|s) = 1$ ，当 $R(\cdot)$恒为正或恒为负时，理论上不会影响学习。

但，如果 $R(\cdot)$ 恒正，可能会陷入一些一开始没被sample到的action，在更新梯度后更不易被sample到的窘境。

可以考虑添加baseline，设计超参 $b$ 并修正为上式为：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left(R(\tau^n) - b \right) \nabla \log(a_t^n|s_t^n, \theta)$

事实上，参数 $b$ 可以是常数，也可以与 $s$ 有关。



**更合理的评估reward** (trick)：

对于某一episode $\tau $ 某一时刻 $t$ ，给予整个 $\tau$ 的reward可能是不公平的，因为 $s_t$ 下的任何决策的好坏似乎和 $t$ 时刻前已经积累的reward是无关的。修正上式（维护后缀和）为：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left( \sum_{i=t}^{T} r_i^n - b \right) \nabla \log(a_t^n|s_t^n, \theta)$

更进一步，可以考虑将较 $t$ 时刻过于遥远的reward打一个折扣，设计超参 $\gamma < 1$ ，修正上式为： 

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} A^\theta(s_t, a_t) \nabla \log(a_t^n|s_t^n, \theta)$

其中： $A^\theta(s_t, a_t) = \sum_{i=t}^{T} \gamma^{i-t}r_i^n - b $



**Off-policy** (trick)：

先不考虑优化，我们有 $ \nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau) \nabla\log p_\theta(\tau)]$

值得注意的是，游戏 $\tau$ 发生的概率分布与参数 $\theta$ 是有关的。即我们是用策略 $\pi_\theta$ 去不断采样 $\tau$ 的。

因此，在求梯度时，当参数 $\theta$ 更新后，我们不得不立即用新的 $\theta$ 重新从环境中进行采样 (**on-policy**)。

根据概率论，我们有：

$\displaystyle \mathbb{E}_{x \sim p}[f(x)] = \int {f(x)p(x) } dx = \int f(x)\frac{p(x)}{q(x)}q(x) dx = \mathbb{E}_{x \sim q}\left[f(x)\frac{p(x)}{q(x)} \right]$

因此，我们只需添加修正因子 $\frac{p(x)}{q(x)}$ ，就可以从分布 $q(\tau) = p_{\theta^{old}}(\tau)$ 中得到对 $p(\tau) = p_{\theta^{new}}(\tau)$ 的无偏估计。

可惜的是，虽然两者期望一致，但当 $p(x)$ 和 $q(x)$ 在采取行为的表现差异过大时，估计的方差会显著变大。

因此在不考虑优化的情况下，off-policy有：

 $\displaystyle \nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta^\prime}} \left[ \frac{p_\theta(\tau)}{p_{\theta^\prime}(\tau)} R(\tau) \nabla \log p_\theta(\tau) \right]$

同理，如果采用 $ A^\theta(s_t, a_t) $ 评估reward，则有：

 $\displaystyle \nabla \bar{R}_\theta = \mathbb{E}_{(s_t,a_t) \sim \pi_\theta} \left[ A^\theta(s_t, a_t) \nabla \log(a_t^n|s_t^n, \theta) \right] \\             \displaystyle = \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t,a_t)}{p_{\theta^\prime}(s_t,a_t)} A^\theta(s_t, a_t) \nabla \log(a_t^n|s_t^n, \theta) \right] \\                                                                        \displaystyle = \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t|a_t)}{p_{\theta^\prime}(s_t|a_t)} \frac{p_\theta(s_t)}{p_{\theta^\prime(s_t)}} A^\theta(s_t, a_t) \nabla \log(a_t^n|s_t^n, \theta) \right]$

在 $\theta^\prime \approx \theta$ （行为上而非参数上）时，假设$A^{\theta^\prime}(s_t,a_t) \approx A^\theta (s_t,a_t)$ 和 $p_\theta(s_t) \approx p_{\theta^\prime}(s_t)$ ，则有：

$\displaystyle \nabla \bar{R}_\theta \approx \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t|a_t)}{p_{\theta^\prime}(s_t|a_t)}  A^{\theta^\prime}(s_t, a_t) \nabla \log(a_t^n|s_t^n, \theta) \right]$

PPO/TRPO/PPO2

PPO：Policy-Based



### Q-learning

**Critic**(Value-based)： $V^\pi(s)$ 表征 $\pi$ 策略下 $s$ 的期望reward，critic是与actor绑定的。

State value function  $V^\pi(s)$：

- Monte-Carlo (MC) based approach：更新 $V^\pi(s_t) = \sum_{i\geq t} r_i$ 。至少要等到游戏结束才能更新network，耗费时间长。
- Temporal-difference (TD) approach：估算$V^\pi(s_{t+1})$，再更新 $V^\pi(s_t) = V^\pi(s_{t+1})+r_t$。拥有更小的variance，但 $V^\pi(s_{t+1})$ 不一定估的准。

State-action value function  $Q^\pi(s, a)$：同样可以用TD或MC的方式。



如果对于任意 $s$，定义 $\pi^\prime(s) = \arg\max_a Q^\pi(s,a)$，则有恒等式：

$\displaystyle V^\pi(s_t) = Q^\pi(s_t,\pi(s_t)) \leq \max_a Q^\pi(s_t,a) = Q^\pi(s, \pi^\prime(s_t)) = r_t + V^\pi(s_{t+1}) \\                                        = r_t + Q^\pi(s_{t+1}, \pi(s_{t+1}))\leq r_t + \max_aQ^\pi(s_{t+1}, a) \cdots \leq V^{\pi^\prime}(s) $

因此总可以通过 $Q^\pi$ 得到一个更好的策略 $ \pi^\prime $



**Fixed network (target network)**：

在训练中，我们认为 $r_t + Q^\pi(s_{t+1}, \pi(s_{t+1}))$ 是对 $Q^\pi(s_t, a_t)$ 的一个更精确的估计。我们将前者中的 $Q^\pi$ 固定进行多轮更新得到 $\hat{Q} $，在若干轮后再将 $\hat{Q}$ 替换 $Q$ 进行新一轮迭代。



**Exploration**：

如果每次决策都取 $a = \arg\max_aQ(s,a)$ ，非常不利于对环境的探索。

- Epsilon Greedy：给定一个会随着训练递减的 $\varepsilon$ ，以 $\varepsilon$ 的概率进行随机决策
- Boltzmann Exploration：根据 $Q$ 的概率分布normalize后进行决策 $P(a|s) = \frac{e^{Q(s,a)}}{\sum_a e^{(Q(s,a))}}$



**Replay Buffer**：





### Actor Critic

[WIP]

