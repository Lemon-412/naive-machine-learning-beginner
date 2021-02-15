# Reinforcement Learning



> 参考资料：李宏毅强化学习2020
>
> https://www.bilibili.com/video/BV1UE411G78S
>
> 01.Deep RL
> 02.Policy Gradient
> 03.Learning to Interact with Envs
> 04.PPO
> 05.From on-policy to off-policy
> 06.Q-Learning
> 07.QL继续 DQN
> 08.QL第3段：连续动作的QL
> 09.Actor-Critic
> 10.Sparse Reward
> 11.Imitation Learning
>
> 已完成：p1, p2, p3, p4, p5, p6, p7, p8



### Scenario of RL

RL由 **Agent** 和 **Environment** 组成：

- **State**(Observation)：Agent 观测 Environment。DRL的强表达能力使得可以直接将 Observation 作为State，而不需要做额外的 Summary 。
- **Action**：Agent 产生行为改变 Environment，Environment 给予 **reward**

- Agent learns to take actions to maximize expected reward.



### Learning from experience

RL从环境学习的过程中产生一系列的 Action ，但直到到达终态（完成一个**episode** $\tau$），大多数的 reward 可能都是0，缺少对每一个行为的好坏的评判。因此，相较于 supervised learning ，RL更需要从经验中进行学习（reward delay）。



> Inversed Reinforcement Learning (IRL)
>
> 通常RL是给定 environment ，通过黑盒的 reward function 去学一个 $\theta$ 
>
> IRL用于解决现实中 reward function 复杂甚至无法定义的情况，给定 environment ，并通过 expert 的知道去学习一个 reward function 。
>
> > 如何定义无人驾驶的reward function？撞了人扣一百分，那撞了狗呢？
>
> 具体实现方法类似 GAN ， Actor 始终模仿 expert 的行为（最终成为 generator ），而 Discriminator 始终学着区分 Actor 和 expert （最终成为 reward function ）



### 马尔科夫决策过程对RL的描述

定义环境 $E=<X,A,P,R>$

- 状态空间$X$：对 Agent 感知到的环境的描述。$\forall s \in X$唯一对应一个环境的状态（State）。

- 动作空间$A$：对 Agent 能采取的行为的约束。$\forall a \in A$唯一对应一个可采取的行为。在一些问题中，$a|x$。

- 转移函数$P$：$X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作下转移到另一状态的概率。黑盒。

- 奖赏$R$：$X \times X \rightarrow \mathbb{R}$ 或 $X \times A \times X \rightarrow \mathbb{R}$，描述环境在某一状态某一动作转移到某一状态的奖赏（reward）。黑盒。



### Policy Gradient

定义策略 $\pi$ ：$X \times A \rightarrow \mathbb{R}$。即 $\pi\left( s, a \right)$ 表示状态 $s$ 下执行动作 $a$ 的概率。这是 Actor 的学习目标。

**Actor**(policy)： $\pi _{\theta} \left( s, a \right)$:

- input： State，一般用一个向量或矩阵表示
- output： Action，采取每一个 action 对应的几率



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

所以有：

 $\displaystyle \nabla \log P(\tau|\theta) = \sum_{t=1}^T \nabla \log p(a_t|s_t, \theta)$



所以：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} R(\tau^n) \nabla \log p(a_t^n|s_t^n, \theta)$

即进行N局游戏后，每局游戏每一时刻采取策略的概率与该局游戏总收益的乘积的和的梯度。

注意：收益使用 $\tau^n$ 的总收益而非单步收益。



为什么刻意使用  $\nabla \log P(\tau|\theta)$ 代替 $\nabla P(\tau|\theta)$？

- 将 $\sum_{\tau}P(\tau|\theta)f(\cdot )$ 转换为 $ \frac{1}{N}f(\cdot)$求解
- 将 $P(\tau|\theta)$ 中的连乘转换为连加求解
- 直观上除以 $P(\tau|\theta )$ 使得 $\pi_\theta$ 中不常规的却高收益的行为 $\tau_t$ 有了被大幅更新的可能



**Add a Baseline**：

由于 actor 对各个行为的概率会做归一化处理，即 $\sum_{a} \pi_\theta(a|s) = 1$ ，当 $R(\cdot)$恒为正或恒为负时，理论上不会影响学习。然而，如果 $R(\cdot)$ 恒正，可能会陷入一些一开始没被 sample 到的 action ，在更新梯度后更不易被 sample 到的窘境。

可以考虑添加 baseline ，设计超参 $b$ 并修正为上式为：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left(R(\tau^n) - b \right) \nabla \log p_\theta(a_t^n|s_t^n)$

事实上，参数 $b$ 可以是常数，也可以与 $s$ 有关。



**合理评估 reward**：

对于某一episode $\tau $ 某一时刻 $t$ ，给予整个 $\tau$ 的 reward 可能是不公平的，因为 $s_t$ 下的任何决策的好坏似乎和 $t$ 时刻前已经积累的reward是无关的。修正上式（维护后缀和）为：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left( \sum_{i=t}^{T} r_i^n - b \right) \nabla \log p_\theta(a_t^n|s_t^n)$

更进一步，可以考虑将较 $t$ 时刻过于遥远的 reward 打一个折扣，设计超参 $\gamma < 1$ ，修正上式为： 

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n)$

其中： $A^\theta(s_t, a_t) = \sum_{i=t}^{T} \gamma^{i-t}r_i^n - b $



**Off-policy**：

先不考虑优化，我们有 $ \nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau) \nabla\log p_\theta(\tau)]$

值得注意的是，游戏 $\tau$ 发生的概率分布与参数 $\theta$ 是有关的。即我们是用策略 $\pi_\theta$ 去不断采样 $\tau$ 的。

因此，在求梯度时，当参数 $\theta$ 更新后，我们不得不立即用新的 $\theta$ 重新从环境中进行采样 (**on-policy**)。

根据概率论，我们有：

$\displaystyle \mathbb{E}_{x \sim p}[f(x)] = \int {f(x)p(x) } dx = \int f(x)\frac{p(x)}{q(x)}q(x) dx = \mathbb{E}_{x \sim q}\left[f(x)\frac{p(x)}{q(x)} \right]$

因此，我们只需添加修正因子 $\frac{p(x)}{q(x)}$ ，就可以从分布 $q(\tau) = p_{\theta^{old}}(\tau)$ 中得到对 $p(\tau) = p_{\theta^{new}}(\tau)$ 的无偏估计。

可惜的是，虽然两者期望一致，但当 $p(x)$ 和 $q(x)$ 在采取行为的表现差异过大时，估计的方差会显著变大。

因此在不考虑优化的情况下，off-policy有：

 $\displaystyle \nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta^\prime}} \left[ \frac{p_\theta(\tau)}{p_{\theta^\prime}(\tau)} R(\tau) \nabla \log p_\theta(\tau) \right]$

同理，如果采用 $ A^\theta(s_t, a_t) $ 评估 reward ，则有：

 $\displaystyle \nabla \bar{R}_\theta = \mathbb{E}_{(s_t,a_t) \sim \pi_\theta} \left[ A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n) \right] \\             \displaystyle = \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t,a_t)}{p_{\theta^\prime}(s_t,a_t)} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n) \right] \\                                                                        \displaystyle = \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t|a_t)}{p_{\theta^\prime}(s_t|a_t)} \frac{p_\theta(s_t)}{p_{\theta^\prime(s_t)}} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n) \right]$

在 $\theta^\prime \approx \theta$ （行为上而非参数上）时，假设$A^{\theta^\prime}(s_t,a_t) \approx A^\theta (s_t,a_t)$ 和 $p_\theta(s_t) \approx p_{\theta^\prime}(s_t)$ ，则有：

$\displaystyle \nabla \bar{R}_\theta \approx \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta^\prime}} \left[ \frac{p_\theta(s_t|a_t)}{p_{\theta^\prime}(s_t|a_t)}  A^{\theta^\prime}(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n) \right]$

PPO/TRPO/PPO2

PPO：Policy-Based



### Q-learning

**Critic**(Value-based)：

State value function  $V^\pi(s)$：在 $s$ 下使用 $\pi$ 继续游戏的期望累计收益。

- Monte-Carlo (MC) based approach：更新 $V^\pi(s_t) = \sum_{i\geq t} r_i$ 。至少要等到游戏结束才能更新 network ，耗费时间长。
- Temporal-difference (TD) approach：预测 $V^\pi(s_{t+1})$，再更新 $V^\pi(s_t) \leftarrow V^\pi(s_{t+1})+r_t$。拥有更小的variance，但 $V^\pi(s_{t+1})$ 不一定估的准。

State-action value function  $Q^\pi(s, a)$：在 $s$ 下进行 $a$ ，并使用 $\pi$ 继续游戏的期望累计收益。同样可以用TD或MC的方式。

注意 critic 永远是与 actor ($\pi$)绑定的，而不是对环境的客观评估。



如果对于任意 $s$，定义 $\pi^\prime(s) = \arg\max_a Q^\pi(s,a)$，则在期望条件下，有恒不等式：

$\displaystyle V^\pi(s_t) = Q^\pi(s_t,\pi(s_t)) \leq \max_a Q^\pi(s_t,a) = Q^\pi(s, \pi^\prime(s_t)) = r_t + V^\pi(s_{t+1}) \\                                        = r_t + Q^\pi(s_{t+1}, \pi(s_{t+1}))\leq r_t + \max_aQ^\pi(s_{t+1}, a) = \cdots \leq V^{\pi^\prime}(s_t) $

因此总可以通过 $Q^\pi$ 构造出一个更好的策略 $ \pi^\prime $



**Fixed network (target network)**：

在训练中，我们认为 $r_t + Q^\pi(s_{t+1}, \pi(s_{t+1}))$ 是对 $Q^\pi(s_t, a_t)$ 的一个更精确的估计。我们将前者中的 $Q^\pi$ 固定进行多轮更新得到 $\hat{Q} $，在若干轮后再将 $\hat{Q}$ 替换 $Q$ 进行新一轮迭代。



**Exploration**：

如果每次决策都取 $a = \arg\max_aQ(s,a)$ ，非常不利于对环境的探索。

- Epsilon Greedy：给定一个会随着训练递减的 $\varepsilon$ ，以 $\varepsilon$ 的概率进行随机决策
- Boltzmann Exploration：根据 $Q$ 的概率分布 normalize 后进行决策 $P(a|s) = \frac{e^{Q(s,a)}}{\sum_a e^{(Q(s,a))}}$



**Replay Buffer** (Experience buffer)：

将 Trajectory 拆分成 $<s_t, a_t, r_t, s_{t+1}>$ 的集合存放在 replay buffer 中（就像dataset一样）

每次训练随机挑选一条记录。当buffer存满时，替换最久远的记录。

replay buffer本质是 off-policy 的，但考虑到TD相比于MC对 Trajectory 不敏感，所以即使策略 $\pi$ 已经更新，也不影响 replay buffer 的直接使用



**Typical Q-learning Algorithm**：

- Initialize Q-function $Q$ , fixed Q-function $\hat{Q} \leftarrow Q$
- In each episode
  - for each time-step $t$ :
    - Given state $s_t$ , taken action $a_t$ based on $Q$ (epsilon greedy)
    - Obtain reward $r_t$ and the new state $s_{t+1}$
    - Store  $<s_t, a_t, r_t, s_{t+1}>$ into replay buffer
    - Sample $<s_i, a_i, r_i, s_{i+1}>$ randomly from the buffer
    - let $y \leftarrow r_i + \max_a \hat{Q}(s_{i+1}, a)$
    - Update the parameters of $Q$ to make $Q(s_i, a_i)$ close to $y$
  - meanwhile, for each time-step $c$ :
    - reset $\hat{Q} \leftarrow Q$



**Double DQN**：

在 DQN 中， $ Q(s_t, a_t) \leftarrow r_t + \max_a Q(s_{t+1}, a)$ 很容易导致 $Q(s,a)$ 整体被高估。因为只要 $Q(s_{t+1}, a)$ 中有一个被高估，就会导致 $Q(s_t,a_t)$ 被高估。

DDQN 构建两个 Q-function ，一个负责决策最好的 $a$，一个给出 $<s,a>$ 的期望 reward ，降低高估的可能性。

  $Q(s_t, a_t) \leftarrow r_t + Q^\prime(s_{t+1}, \arg\max_aQ(s_{t+1},a))$

特别的，可以选择让 $Q^\prime = \hat{Q}$



**Dueling DQN**：

改变 DQN 的 network 结构，输入一个 state ，输出一个值 $V(s)$ 和一个向量 $A(s,a)$ 。

有 $Q(s,a) = V(s) + A(s,a)$ 且 $\sum_a(A(s,a)) = 0$

好处：对 state 的好坏进行评估，有机会在一次训练中更新多个 $Q$ 值，提升效率。



**Prioritized Replay**：

TD error 大的 Experience 有更大的可能被 sample 到。 

update network 的方式应当随之改变。



**Multi-step**：

multi-step 是MC和TD的折衷方案。

replay buffer 存放 $<s_t, r_t, r_{ty+1}, \cdots, r_{t+N}, s_{t+N}>$

使用 $\sum_{t^\prime = t}^{t+N}r_{t^\prime}+\hat{Q}(s_{t+N+1},a_{t+N+1})$ 进行更新。



**Noisy Net**：

Epsilon Greedy 为了更好的探索环境，选择在 Action 上添加 noise 。

noisy net 选择在 Q-function 上添加噪音（如高斯噪音）以达到探索的目的。

$a = \arg\max_a \tilde{Q}$ ，其中 $\tilde{Q}$ 由 $Q$ 添加噪音得到。 $\tilde{Q}$ 生存周期可以是一个 trajectory。

Noisy net 保证了在一个 episode 中，同一个 state 的 action 永远是一致的（state-dependent exploration）。



**Distributional Q-function**：

$Q(s, a)$ 的本质是在 $s$ 下采取 $a$ 的期望累积收益，本质应该是一个分布而不止包含均值。

Distributional Q-function 的 network 在输入 $s$ 后，会输出对应的每一个 $a$ 的期望收益的概率分布。



**Rainbow**：

DQN + DDQN + Prioritized DDQN + Dueling DDQN + A3C(multi-step) +Distributional DQN + Noisy DQN

凑齐彩虹，召唤神龙！



**Continuous Actions**:

如果 action 是连续的，就不能直接通过穷举找到 $\arg\max_a Q(s,a)$

- 可以尝试将 action 随机采样代入。
- 可以尝试使用梯度上升。
- 可以尝试改变 network 结构。输入 $s$ 输出一个列向量 $\mu(s)$ 、一个矩阵  $\Sigma(s)$ 、一个值 $V(s)$，且满足 $Q(s,a) = -(a-\mu(s))^\top \Sigma(s)(a-\mu(s)) + V(s)$ 。由于 $(a-\mu(s))^\top \Sigma(s)(a-\mu(s))$ 恒正，所以有 $\mu(s) = \arg\max_aQ(s,a)$ 

- 别用 critic 直接决策行为，再炼一个 actor。



### Actor Critic

在 policy gradient 中，我们有：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t^n|s_t^n)$

并使用如下方式估算 $A^\theta$ ： $A^\theta(s_t, a_t) = \sum_{i=t}^{T} \gamma^{i-t}r_i^n - b $

然而 $A^\theta$ 是具有随机性的，本质上应该取足够次数进行 sample，或者直接使用期望。



在 Q-learning 中，我们有：

- State value function  $V^\pi(s)$：在 $s$ 下使用 $\pi$ 继续游戏的期望累计收益。输入 $s$ ，输出一个 value。
- State-action value function  $Q^\pi(s, a)$：在 $s$ 下进行 $a$ ，并使用 $\pi$ 继续游戏的期望累计收益。输入 $s$ ，输出每一个 $a$ 对应的 value。



**Actor-Critic**：

考虑将上述两者结合，则有：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left( Q^{\pi_\theta}(s_t^n,a_t^n) - V^{\pi_\theta}(s_t^n) \right) \nabla \log p_\theta(a_t^n|s_t^n)$



**Advantage Actor-Critic**：

观察到： $\displaystyle Q^\pi(s_t, a_t) =  \mathbb{E} \left[ r_t + V^\pi(s_{t+1}) \right] \approx r_t + V^\pi(s_{t+1})$

为了减少训练的网络数量，我们考虑使用 $V$ 代替 $Q$，则有：

$\displaystyle \nabla\bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T^n} \left( r_t^n + V^{\pi_\theta}(s_{t+1}^n) - V^{\pi_\theta}(s_t^n) \right) \nabla \log p_\theta(a_t^n|s_t^n)$

同时，上式中的 $r_t$ 的 variance 依然比原式中 $A^\theta$ 更小，加强了训练的稳定性。

考虑到 actor $\pi(s)$ 和 critic $V(s)$ 输入是一致的，可以将两个网络放在一起直接训练。输出分别是 $V^\pi(s)$ 和各个 action 的 possibility。对 $s$ 很大的情况（如图像）尤为实用。



**Asynchronous Advantage Actor-Critic**：

影响RL训练时间的瓶颈往往是与 environment 交互取得数据的过程，为了加快这一过程可以考虑并行采样。

Asynchronous Advantage Actor-Critic会使用一个 global network（包含 actor 和 critic），和若干个 worker 。

每个 worker 会不断地从 global network 中拷贝参数，并与环境交互取得 sample data ，计算梯度后回传 global network。