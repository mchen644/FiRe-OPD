# Adaptive Concise-Probe OPD 的 Budget Accounting

日期：2026-07-17  
状态：设计分析；尚未确定最终采用 raw-token budget 还是实际 GPU compute budget

## 1. 目标

本项目的核心仍然是 reasoning compression，而不是减少训练中见到的题目数。

每个训练 step 应保持与 Vanilla OPD 相同的题目覆盖：

- 每个 batch 包含 1,024 道不同题目；
- 50 steps 共看到约 51,200 道题；
- 不再使用 256 道题、每题 rollout 4 次来模拟 1,024 trajectories，因为这会把题目覆盖率降低到 Vanilla 的四分之一。

真正需要判断的不是抽象的绝对难度，而是：**一道题是否能够安全地使用 concise compression。**

## 2. Adaptive asymmetric two-prompt routing

对每道题先生成一条 student normal-prompt rollout。

- normal rollout 错误：直接归为 hard/non-easy，不再生成第二条 rollout；
- normal rollout 正确：额外生成一条 concise-prompt diagnostic rollout。

路由如下：

| Normal rollout | Concise rollout | 分类 | Teacher prompt | Normal rollout supervision |
|---|---|---|---|---|
| 错误 | 不生成 | hard / non-easy | normal | 100% |
| 正确 | 错误 | compression-sensitive / learnable | normal | 100% |
| 正确 | 正确 | compression-safe easy | concise | 压缩后的前缀 |

真正进入 OPD 的始终是第一条 normal rollout：

- concise rollout 只用于 reward 和路由；
- concise rollout 不进入 old-log-prob、teacher/ref log-prob 或 actor update；
- easy 样本由 concise teacher prompt 在 normal rollout token 上提供 distillation signal；
- learnable 和 hard 样本保留 normal teacher prompt 与完整 supervision。

这种定义直接测量“是否可压缩”，并且避免 best-of-two trajectory selection bias。

## 3. 逐题 token accounting

记：

- \(L_n\)：normal rollout 长度；
- \(L_c\)：concise diagnostic rollout 长度。

最直接的 easy 路径可以设置为：

\[
L_{\text{sup}} = \max(0, L_n - L_c).
\]

对于 easy 样本，新增的 \(L_c\) 个 concise-generation token，正好由 normal rollout 中少监督的 \(L_c\) 个 token 抵消。

但这不能保证整个 batch 严格零增量。对于 learnable 样本：

- 已额外生成 \(L_c\) 个 concise token；
- 又保留了 100% 的 normal supervision。

因此，相对 Vanilla 的 raw-token 增量为：

\[
\Delta B = \sum_{i\in\text{learnable}} L_{c,i},
\]

另外还要包含少数 \(L_c > L_n\) 时无法完全抵消的部分。

所以，逐题使用 \(L_n-L_c\) 可以覆盖 easy diagnostic 的成本，但不能严格覆盖失败的 concise probes。

## 4. 为什么实际增量可能仍然很小

举例假设：

- normal 正确率为 70%；
- normal 正确的题中，80% 在 concise prompt 下也正确；
- concise rollout 平均长度为 normal rollout 的 50%。

那么：

- concise diagnostic 新增约 \(0.7\times0.5=35\%\) 的 response tokens；
- easy supervision 回收约 \(0.7\times0.8\times0.5=28\%\)；
- 净 raw-token 增量约为 7%。

现有 1,024-question、n=1 日志中，每 step 大致为：

- student generation：约 846 秒；
- old-log-prob + teacher/ref + actor update：约 826 秒；
- 总时间：约 1,681 秒。

因此，如果 easy normal responses 在进入 old-log-prob、teacher/ref 和 actor update 前被物理截短，那么一个新增 generation token 与一个被移除的 post-rollout distillation/update token，在当前系统中的成本量级接近。

这意味着该方案更可能带来个位数到十几个百分点的 overhead，而不是固定 n=2 路径的 30%–45%。不过这必须通过 1–3 step profiling 验证，不能仅凭 token 数宣称实际 compute 完全相同。

## 5. 严格 batch-global budget controller

如果希望严格声明 budget-neutral，可以在 batch 级别统一结算，而不是逐题各自结算。

先计算全部 diagnostic concise rollouts 的总长度：

\[
D = \sum_{i\in\text{normal-correct}} L_{c,i}.
\]

然后：

1. learnable 和 hard 样本继续保留 100% normal supervision；
2. 只从 easy 样本的 normal supervision 中回收 token；
3. 使 easy 样本总共移除的 supervision token 满足：

\[
\sum_{i\in E}(L_{n,i}-L_{\text{sup},i}) = D.
\]

可以通过 water-filling 分配回收量，并为每个 easy 样本设置安全下限，例如：

- 至少保留 normal rollout 的 20%；或
- 至少保留 256 tokens。

如果 easy 样本没有足够的可回收容量，则必须记录 `budget_overflow`，不能宣称严格 budget-neutral。

## 6. Raw-token budget 与实际 compute budget

“token 数相同”不自动等于“GPU compute 相同”，因为：

- student rollout 是 autoregressive decoding；
- teacher scoring 是对固定 response 的并行 forward；
- old-log-prob、teacher/ref 和 actor backward 的每 token 成本不同；
- 短序列可能降低 batching 与 packing 效率。

因此有两个可选定义：

### A. Raw-token-neutral

把新增 concise-generation tokens 与移除的 supervision tokens 按 1:1 计算。

优点：简单、透明、容易实现。  
缺点：只能声明 token budget 近似不变，不能严格声明 FLOPs 或 wall time 不变。

### B. Compute-neutral

先通过短 profiling 标定 exchange rate：

\[
\rho = \frac{\text{student concise generation cost/token}}
{\text{old+ref+actor cost/supervised token}}.
\]

然后要求 easy 路径回收：

\[
\sum_{i\in E}(L_{n,i}-L_{\text{sup},i})
= \rho D.
\]

优点：可以更有力地声明实际 compute budget 近似持平。  
缺点：需要固定硬件、实现和 profiling contract，并处理 packing efficiency 的非线性。

## 7. 建议的论文表述

在没有实际 profiling 之前，不应直接声称“没有增加训练成本”。更稳妥的表述是：

> We reallocate post-rollout distillation compute from compression-safe examples to adaptive concise diagnostic rollouts, while preserving full supervision for compression-sensitive and hard examples.

完成 wall-time/FLOP 标定并满足 batch-global controller 后，才适合进一步声明 compute-neutral。

## 8. 待确认决策

最终需要固定 budget 的定义：

1. raw response-token budget；或
2. 实际 GPU wall-time/FLOP budget。

若目标是形成有说服力的 compute-neutral claim，推荐第二种，并先运行 1–3 step calibration profile。
