# Difficulty-Routed Budget20 OPD 当前策略说明

日期：2026-07-07

本文记录当前正在跑的 Difficulty-Routed Budget20 OPD 策略、实际运行配置、观察到的压缩强度，以及为什么现在可能“压得太死”。

## 1. 当前实验

当前 run：

```text
opd-budget20-da-hardentropy-step50-noprobe-compileoff-20260706_210724
```

关键日志：

```text
logs/difficulty_routed/opd-budget20-da-hardentropy-step50-noprobe-compileoff-20260706_210724.log
```

关键设置：

```text
student: Qwen3-4B
teacher/ref: Qwen3-30B-A3B-Instruct-2507
rollout_n: 1
student rollout prompt: normal/raw prompt
teacher prompt: per-sample routed budget/concise prompt
rethinking probe: disabled
torch compile: disabled
hard entropy: enabled, coef=0.001
total steps: 50
save at step: 50
```

## 2. 目标

当前策略的目标是：

1. 继续用强 teacher 的 OPD 信号提升 student reasoning。
2. 用 Budget/ESR 减少过监督，避免 student 学到过长、过啰嗦的 teacher 轨迹。
3. 对“容易样本”更强压缩，对“困难样本”少压缩并保留探索。
4. 在不额外采样多条 rollout 的情况下，用单条 rollout 的 correctness + confidence 近似难度。

## 3. 难度路由信号

当前使用单 rollout 的两信号难度估计：

```text
correct_i = 1[reward_i > 0.5]
confidence_rank_i = percentile_rank(mean_old_log_prob_i within batch)

easy_i = correct_i * confidence_rank_i
hard_i = (1 - correct_i) * (1 - confidence_rank_i)
```

含义：

- 正确且 student 自己高置信：更像 easy。
- 错误且 student 自己低置信：更像 hard。
- 正确但低置信、错误但高置信：属于不确定/混合区域。

当前不是用 confidence 本身当 difficulty，而是用 confidence 去修饰 correctness/wrongness。

## 4. Teacher prompt 路由

每个样本根据 `easy_i` 选择 teacher prompt：

```text
if easy_i >= 0.7:
    teacher prompt = concise
else:
    teacher prompt = budget
```

当前默认：

```text
easy_prompt_threshold = 0.7
easy_prompt_style = concise
default_prompt_style = budget
```

budget teacher prompt 使用当前 student rollout 长度作为预算：

```text
budget_i = round(alpha * rollout_response_length_i)
alpha = 1.0
```

然后 teacher prompt 类似：

```text
Let's think step by step and use less than {budget_i} tokens.
```

concise teacher prompt 类似：

```text
Solve concisely. Avoid unnecessary explanation.
```

## 5. ESR / hard-truncation 策略

当前使用 rollout length 作为 TALE budget 来源：

```text
TALE_BUDGET_SOURCE = rollout_length
TALE_ROLLOUT_ALPHA = 1.0
TALE_ESR_BETA = 0.2
TALE_TRUNCATE_TO_ESR = True
```

基础 ESR 监督比例是 20%。难度路由会按 easy 程度进一步降低 easy 样本的 ESR beta：

```text
base_beta = 0.20
esr_beta_i = clamp(base_beta - easy_esr_delta * easy_i,
                   min=min_easy_esr_beta,
                   max=base_beta)

easy_esr_delta = 0.10
min_easy_esr_beta = 0.10
```

所以：

- hard / non-easy 样本大多接近 20% 监督。
- easy 且高置信正确样本可以降到 10% 监督。
- 所有样本都会被 ESR mask 限制，只监督前缀。

实现语义：

```text
tale_budget/response_length_mean = 原始 rollout 长度
response_length/mean = 实际用于训练的 ESR hard-truncated 长度
```

也就是说，当前日志里看到的 `response_length/mean` 不是原始回答长度，而是监督宽度。

## 6. Hard entropy

当前 hard entropy 已重新打开：

```text
hard_entropy_coef = 0.001
entropy_weight_i = hard_i * hard_entropy_coef
```

它只对 hard-ish 样本加探索项，目标是避免错误低置信样本过早塌缩。

注意：hard entropy 不能补回被 ESR 截掉的后缀监督。它只是在当前监督窗口内鼓励更高 entropy。

## 7. 当前运行中的实际压缩强度

当前 run 后期几步观察值大致是：

```text
correct_rate:                      ~0.66
concise_prompt_ratio:              ~0.24
budget_prompt_ratio:               ~0.76
orig_response_length_mean:         ~1850 tokens
response_length/mean:              ~320 tokens
tale_budget/esr_supervised_fraction_mean: ~0.164
hard_entropy_weight_mean:          ~0.00020
```

解释：

- 原始 rollout 平均约 1800+ tokens。
- 实际训练只看约 320 tokens。
- 平均只监督约 16% 的回答前缀。
- 约 1/4 样本走 concise teacher，约 3/4 仍走 budget teacher。

这说明当前策略的主要压缩来自 ESR hard truncation，而不是 concise prompt 比例本身。

## 8. 为什么现在可能压得太死

当前配置偏激进，原因有三点：

### 8.1 所有样本都被限制到 10%-20% 前缀监督

即使是 hard/wrong 样本，也通常只有 20% 左右的 prefix supervision。

这可能导致模型学到“开头像 reasoning”，但缺少中后段推导、case split、纠错和 finalization 的 teacher 信号。

### 8.2 easy 样本被双重压缩

easy 样本同时受到：

1. concise teacher prompt；
2. ESR beta 从 0.20 降到最多 0.10；
3. hard truncation 只保留前缀。

如果 easy 判定有噪声，正确但其实需要完整推导的样本可能被过早压短。

### 8.3 budget prompt 不等于训练监督长度

budget prompt 的 token budget 约等于原始 rollout length，但训练时又只监督 `beta * budget` 的前缀。

因此表面是 Budget20，实际训练信号更像：

```text
budget prompt + 10%-20% prefix-only OPD
```

这比单纯“让 teacher 简洁一些”更强。

## 9. 当前策略一句话总结

当前策略是：

> student 正常 rollout；用 reward correctness + student old-logprob rank 判断 easy/hard；easy 样本使用 concise teacher 并把 ESR 降到 10%-20%；其它样本使用 rollout-length budget teacher 并保留约 20% 前缀监督；hard/wrong-low-confidence 样本额外加小 hard entropy。

实际效果上，它是一个比较强的 prefix-compression OPD，而不是温和的 prompt-only 压缩。

## 10. 如果要放松，优先调哪些旋钮

如果确认“压得太死”，建议优先按保守顺序放松：

### 方案 A：先把 ESR 整体放宽

推荐作为下一轮最小改动：

```text
TALE_ESR_BETA = 0.30
DA_MIN_EASY_ESR_BETA = 0.15
DA_EASY_ESR_DELTA = 0.05
```

预期：

- hard/non-easy 样本从 20% 监督提高到约 30%。
- easy 样本最低约 15%，不是 10%。
- 仍保留 difficulty routing 和 concise prompt。

### 方案 B：保护 hard 样本，easy 继续压

思路：让 hard/wrong-low-confidence 样本拿到更长监督窗口，例如 30%-40%，easy 样本再压。

当前代码还没有单独的 `hard_esr_beta`，可以先用更高 base beta 间接实现：

```text
TALE_ESR_BETA = 0.35
DA_EASY_ESR_DELTA = 0.15
DA_MIN_EASY_ESR_BETA = 0.15
```

这样 hard 接近 35%，easy 可降到约 15%-20%。

### 方案 C：先弱化 hard truncation，只保留 prompt 压缩

如果怀疑 prefix-only supervision 是主要伤害，可以测试：

```text
TALE_ESR_BETA = 0.50
DA_MIN_EASY_ESR_BETA = 0.25
DA_EASY_ESR_DELTA = 0.10
```

这会明显增加训练宽度，但更安全。

## 11. 当前建议

不要立刻把所有机制都关掉。更合理的下一步是：

1. 保留 difficulty-routed prompt。
2. 保留 hard entropy，但可以后续比较 `0.001` vs `0.0005`。
3. 先放宽 ESR，因为当前平均监督比例只有约 16%。
4. 下一轮目标让 `response_length/mean` 从 ~320 提到约 450-600，观察 hard math accuracy 是否恢复。

推荐下一轮从方案 A 开始。它最小改动、保留当前策略结构，但不会把 easy 样本压到 10%、hard 样本压到 20% 那么死。
