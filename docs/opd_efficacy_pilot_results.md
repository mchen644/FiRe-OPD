# Vanilla-OPD Proxy-Gradient Efficacy Pilot 结果报告

## 1. 执行摘要

本次一次性 `efficacy_pilot` 已完成全部 14 个 work units，并通过 complete-stage validation。Canonical report 给出的结论为：

```text
classification status:  pilot_only
operational decision:   no_go
main Stage-1 hypothesis: not_evaluated
```

`pilot_only` 表示该报告只用于本次成本 gate，不能替代完整 Stage-1 结论。

预注册 gate 只使用 primary-K、uniform-null percentiles。K-means seed 42 的 G-Vendi percentile 为 **0.0678**，coverage percentile 为 **0.4591**；两者均不高于 `no_go` 阈值 0.60，因此本 pilot 按预注册规则得到 `no_go`。

该结论是一个**实验成本闸门**：当前 pilot 不支持继续支付完整 Stage 1 的计算成本。它不构成对主假设的否定，不评估 downstream/OOD accuracy，也不证明 proxy-gradient selection 在其他数据、模型、随机种子或训练设置下普遍无效。Stage 1 未启动并保持暂停。

## 2. 实验目标与边界

### 2.1 目标

Pilot 检验下列 proxy-gradient selection 是否在冻结的 target-gradient space 中优于随机子集：

- **Proxy gradient**：冻结 `Qwen3-4B -> Qwen3-0.6B`，表示为 `P_pilot`；
- **Target gradient**：冻结 `Qwen3-30B-A3B-Instruct-2507 -> Qwen3-4B`，表示为 `T_pilot`；
- **候选选择**：只根据 `P_pilot` 选择 56 个 candidates；
- **目标空间评估**：只在 `T_pilot` 中与冻结随机 null schedules 比较。

Pilot 成员严格来自冻结 Stage-1 manifest 的有序前缀：前 250 个 candidate rows 和前 84 个 held-out rows。它不重新采样、不打乱成员、不使用 Stage-1 GPU artifacts。

### 2.2 明确不评估的内容

本实验不评估：

- downstream 或 OOD accuracy；
- SFT、embedding 或 `P_n4` baselines；
- generation seed 43；
- target oracle；
- cross-seed oracle；
- target-seed dependence；
- 完整 Stage-1 主假设。

因此 canonical report 将主假设标记为 `not_evaluated`，并将 cross-seed oracle 与 target-seed dependence 明确标记为 `unavailable`。

## 3. 冻结实验配置

| 项目 | 冻结值 |
|---|---:|
| Candidates | 250 |
| Held-out | 84 |
| Target capture rows | 334 |
| Proxy capture rows | 250 |
| Selected size | 56 |
| Primary K | 25 |
| Diagnostic K | 3 |
| K-means seeds | 42、43 |
| Round-robin seed | 42 |
| Uniform null draws | 10,000 |
| Stratified null draws | 10,000 |
| Null `SeedSequence` | 2026071402 |
| Generation seeds | `[42]` |
| Native `rollout.n` | 1 |
| Representations | `P_pilot`、`T_pilot` |
| TRAK projection dimension | 1,024 |
| Projection seed | 0 |

相对 Stage 1，只允许两项算法偏差：native `rollout.n=1` 和 generation seed set `[42]`。其余关键语义保持冻结：

- maximum response length 16,384；
- Vanilla OPD；
- only reverse-KL advantages；
- token-mean aggregation；
- token-level importance sampling；
- IS upper threshold 5.0；
- micro-batch size per GPU 1；
- 无 optimizer construction/step；
- 无 clipping；
- 无 parameter update。

## 4. 预注册判定规则

Gate 只读取两个 K-means seeds 在 **primary K=25、uniform null** 下的 G-Vendi、coverage、gradient norm 和 OPD signal percentiles。

### `go`

两个 K-means seeds 必须同时满足：

```text
G-Vendi >= 0.90
coverage >= 0.90
gradient norm >= 0.25
OPD signal >= 0.25
```

### `no_go`

只要任一 K-means seed 满足以下任一条件，即判为 `no_go`：

```text
G-Vendi <= 0.60
或
coverage <= 0.60
```

### `borderline`

不满足 `go`，也未触发 `no_go` 时，判为 `borderline`。

这些不等式为 inclusive thresholds。Diagnostic K 和 stratified nulls 只作补充诊断，不能改变 gate。Stage-1 thresholds 未因 pilot 结果而修改。

## 5. 执行与完整性

### 5.1 Work-unit 完成情况

| 阶段 | Work units | 状态 |
|---|---:|---|
| Target/proxy capture | 2 | 完成 |
| Target replay shards | 4 | 完成 |
| Proxy replay shards | 4 | 完成 |
| Vector validation | 1 | 完成 |
| Selection | 1 | 完成 |
| Analysis-input preparation | 1 | 完成 |
| Analysis | 1 | 完成 |
| **总计** | **14** | **14/14 完成** |

Target capture 严格发布 334 个 trajectories，proxy capture 严格发布 250 个 trajectories；两者的 `sampling_args.n` 均为 1。Target actor rank 的持久化 trajectory counts 为 `84 / 84 / 84 / 82`，总数仍为 334。

由于 334 和 250 不能被四个 FSDP ranks 整除，运行时只在 actor/reference RPC transport 内进行最小 padding：target `334 -> 336`，proxy `250 -> 252`。Padding rows 不进入 generation membership、trainer boundary、actor chunks、rank provenance、compound-key coverage、replay vectors 或 completion manifests；controller 在 finalization 前恢复原顺序和原基数。

### 5.2 验证证据

最终 pilot source 的验证结果为：

- non-GPU `math_eval`：611 passed，1 deselected；
- focused VERL：78 passed；
- 14/14 work-unit contracts 与 completion ledgers 匹配；
- complete-stage validation passed；
- Stage-0 completion artifact 保持不变；
- frozen Stage-1 manifest 保持不变；
- Stage-1 runtime work units 与 capture files 均为 0。

## 6. 主要结果

### 6.1 Primary-K uniform-null percentiles（唯一 gate 输入）

| K-means seed | G-Vendi | Coverage | Gradient norm | OPD signal |
|---:|---:|---:|---:|---:|
| 42 | **0.0678** | **0.4591** | 0.0839 | 0.2453 |
| 43 | 0.8534 | 0.7928 | 0.5084 | 0.7568 |

### 6.2 Primary-K target-space raw scores

| K-means seed | G-Vendi | Coverage | Full gradient norm | OPD signal RMS | Response length | Sampled reverse-KL |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 45.843365 | 0.205583 | 16.219798 | 1.424723 | 1736.178571 | 0.376632 |
| 43 | 48.358405 | 0.215294 | 17.598863 | 1.463175 | 1465.089286 | 0.394999 |

### 6.3 Primary-K stratified-null percentiles（非 gate）

| K-means seed | G-Vendi | Coverage | Gradient norm | OPD signal | Response length | Sampled reverse-KL |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.0011 | 0.4101 | 0.0430 | 0.1403 | 0.9816 | 0.5428 |
| 43 | 0.6487 | 0.7574 | 0.4560 | 0.6873 | 0.2870 | 0.9612 |

### 6.4 Diagnostic-K sensitivity（非 gate）

Diagnostic K=3 的核心 percentiles 如下。它们只用于敏感性诊断，不能覆盖 primary-K gate。

| Null | K-means seed | G-Vendi | Coverage | Gradient norm | OPD signal |
|---|---:|---:|---:|---:|---:|
| Uniform | 42 | 0.8006 | 0.1052 | 0.2161 | 0.5517 |
| Uniform | 43 | 0.7488 | 0.6396 | 0.5573 | 0.7720 |
| Stratified | 42 | 0.5546 | 0.1043 | 0.1469 | 0.4404 |
| Stratified | 43 | 0.4744 | 0.5673 | 0.5111 | 0.7077 |

## 7. Gate 判定

判定过程是确定性的：

1. Gate 只读取第 6.1 节的 primary-K uniform-null percentiles；
2. K-means seed 42 的 G-Vendi percentile 为 0.0678，满足 `0.0678 <= 0.60`；
3. 同一 seed 的 coverage percentile 为 0.4591，也满足 `0.4591 <= 0.60`；
4. 任一 seed 的 G-Vendi 或 coverage 不高于 0.60 即触发 `no_go`；
5. 因此最终 operational decision 为 **`no_go`**。

K-means seed 43 的 norm 和 signal 高于最低要求，但 G-Vendi=0.8534、coverage=0.7928 仍低于 `go` 所需的 0.90。更重要的是，预注册规则不允许一个 seed 的结果覆盖另一个 seed 已触发的 `no_go`。

## 8. 结果解释与限制

本结果支持的最窄结论是：在本次冻结 membership、单 generation seed、单 rollout、两组 K-means initialization 和预注册 null schedules 下，proxy-selected subset 未满足继续扩展至完整 Stage 1 的成本 gate。

不得从本 pilot 推导：

- 对主 Stage-1 假设的否定；
- proxy gradients 与 target gradients 普遍不一致；
- 该方法不能提升任何 downstream/OOD accuracy；
- seed 43 的较高 percentiles 构成独立通过；
- diagnostic K 或 stratified-null 结果可以替代 primary uniform gate。

两个 K-means seeds 的结果差异明显，说明 selection 对聚类初始化存在敏感性；预注册 gate 正是通过要求两个 seeds 均达到高阈值来避免依据单个有利 initialization 扩展实验。由于 pilot 只有一个 generation seed 和每题一个 rollout，cross-seed oracle 与 target-seed dependence 均保持 `unavailable`。

## 9. 执行说明

GPU 执行过程中有两个 attempt 按设计 fail-closed：

1. 首次 attempt 在 capture validation 阶段暴露 334 trajectories 不能平均分发到四个 FSDP ranks；随后加入只用于内部 transport 的最小 padding，并验证 padding 不会持久化。
2. 第二次 attempt 在 target capture finalization 阶段暴露遗留的四-slot 常量；随后将 actor key validation、slot coverage 和 sampling-`n` validation 全部绑定到 typed `native_rollouts`。

每次 source 修正都产生新的 source identity；旧 attempt 的 partial artifacts 未被最终运行复用。失败证据保存在：

```text
logs/opd_proxy_gradient_verify/efficacy_pilot/failed_attempts/
```

最终成功运行使用 source commit `020ef80f4a726ec71df496e3db49b223c1f5f568`。

## 10. Artifacts 与 provenance

### 10.1 Canonical paths

```text
data/opd_proxy_gradient_verify/efficacy_pilot/manifest.json
data/opd_proxy_gradient_verify/efficacy_pilot/report.json
data/opd_proxy_gradient_verify/efficacy_pilot/report.md
data/opd_proxy_gradient_verify/efficacy_pilot/STAGE_COMPLETE.json
data/opd_proxy_gradient_verify/efficacy_pilot/capture/
data/opd_proxy_gradient_verify/efficacy_pilot/replay/
data/opd_proxy_gradient_verify/efficacy_pilot/vectors/
data/opd_proxy_gradient_verify/efficacy_pilot/selection/
data/opd_proxy_gradient_verify/efficacy_pilot/work_units/
```

### 10.2 Frozen identities and hashes

| Artifact / identity | SHA / commit |
|---|---|
| Final pilot source commit | `020ef80f4a726ec71df496e3db49b223c1f5f568` |
| Pilot manifest | `a2783e7d95931c7bccd7ff6b8ad0b601977c75402c299b8b3b49b86352abd3d5` |
| Algorithm contract | `d52655ce968726a5249c13d549a75a7fec25bae99a650b0a04a9958262847e28` |
| Sample manifest | `14fd7f4a656a62b3afb50792b2a9795684bb3c51702ce5b8a8a69242151c1e48` |
| Source snapshot logical manifest | `c0d7848c92dd96f8fe86e252eadf64afa27c58003a3aac25c0e4e8642dbed648` |
| Source snapshot file | `183a729809df1c8df6a4d7472958c9c04aa195824fd5376cf0af3140eff67b03` |
| `report.json` | `2bdc1de2d76c403f12cb5b707c86f5baea541492115877bea3720a9e1be4d132` |
| `report.md` | `eff35625227b5ee3406912d9e61206712812c858ea1ed92b345fb2355872534a` |
| `STAGE_COMPLETE.json` | `16577155d939168ea738de022d976588f150e919ce2b783b67fcef7fcb83f22b` |
| Frozen parent Stage-1 manifest | `6d698d75995c777d6faaf1abd385a758977ca0350ce862284f1e9d8751eddd83` |
| Stage-0 completion | `ef1948b4a1b91f74d203e32d9bd4247ea664386cc569113a2b3a5f8079fd6636` |
| Reference commit | `d9484cd3b5991030b901ac4a3a9e2472dbfac2ad` |
| Reference tree | `a0079d8c5e15cb18bb4790f99c43cc19bb9ecd50` |

## 11. 后续处置

依据 `no_go` 成本闸门：

1. Stage 1 保持暂停；
2. 不启动任何 Stage-1 GPU work unit；
3. 不使用 pilot artifacts 满足 Stage-1 resume；
4. 不修改 Stage-1 contract、thresholds 或 frozen manifest；
5. 不将本结果表述为 downstream/OOD 结论或对主假设的否定。

如果未来决定偏离 `no_go` 仍执行 Stage 1，必须获得新的明确授权，并切换回冻结的 Stage-1 source `opd-proxy-gradient-verify-impl@174849613a9c61445765f0b913174d286f3819e4`；pilot artifacts 仍不得复用。
