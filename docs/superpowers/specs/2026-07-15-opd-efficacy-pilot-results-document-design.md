# OPD Efficacy Pilot 结果文档设计

## 目标

新增一份独立、正式、中文的 efficacy pilot 结果报告：

```text
docs/opd_efficacy_pilot_results.md
```

文档面向需要判断是否继续 Stage 1 的项目成员。读者无需阅读实现计划、运行日志或完整 JSON 报告，也能理解实验边界、预注册 gate、观测指标、`no_go` 推导以及后续处置。

## 信息来源

文档只引用已完成并通过 complete-stage validation 的 canonical artifacts：

- `data/opd_proxy_gradient_verify/efficacy_pilot/report.json`
- `data/opd_proxy_gradient_verify/efficacy_pilot/report.md`
- `data/opd_proxy_gradient_verify/efficacy_pilot/STAGE_COMPLETE.json`
- `data/opd_proxy_gradient_verify/efficacy_pilot/manifest.json`
- 对应 capture manifests、work-unit completion ledgers 和 Git provenance

不得重新计算指标、改变 gate、引入 Stage-1 数据，或根据结果追加事后分析阈值。

## 文档结构

1. **执行摘要**：首先给出 `pilot_only / no_go`，并声明主 Stage-1 假设未评估。
2. **实验目标与边界**：说明 pilot 只比较 proxy-selected subset 在 target-gradient space 中相对随机子集的位置，不测试 downstream/OOD accuracy。
3. **冻结配置**：记录 250 candidates、84 held-out、selected size 56、K=25/3、两个 K-means seeds、10,000 个 null draws、generation seed 42、native `n=1`、`P_pilot` 与 `T_pilot`。
4. **预注册 gate**：逐字表达 inclusive `go / no_go / borderline` 条件。
5. **执行与完整性**：记录 14/14 work units、334/250 capture counts、四个 target 与四个 proxy replay shards、完整验证和 Stage-0/Stage-1 不变性。
6. **主要结果**：用表格列出两个 K-means seeds 的 G-Vendi、coverage、gradient norm 和 OPD signal uniform-null percentiles。
7. **判定推导**：明确 seed 42 的 G-Vendi=0.0678 和 coverage=0.4591 均不高于 0.60，因此触发 `no_go`；seed 43 的较好结果不能覆盖任一 seed 触发的 `no_go` 条件。
8. **解释与限制**：说明结果是成本闸门，而不是主假设失败；cross-seed oracle 和 target-seed dependence unavailable；Stage-1 thresholds 未修改。
9. **产物与 provenance**：列出 source commit、manifest/report/completion SHA、algorithm-contract SHA、parent Stage-1 manifest SHA 和关键路径。
10. **后续处置**：Stage 1 保持暂停，不启动或复用 pilot artifacts。

## 执行异常的呈现范围

正文只简要说明运行中两次 fail-closed：一次是非整除 FSDP dispatch，另一次是遗留四-slot finalizer 常量。说明两者均经测试修复、source identity 更新，并且 partial artifacts 未被复用。详细日志和失败 attempt inventory 只作为 provenance 路径列出，不展开事故复盘。

## 写作约束

- 使用准确、克制的技术语言，不把 percentile 描述为 accuracy 或 downstream gain。
- 始终使用 `no_go`、`pilot_only` 和 `not_evaluated` 的规范拼写。
- 将 percentile 保留四位小数，与 canonical report 一致。
- 不声称 proxy-gradient selection 普遍无效，只说明本 pilot 按预注册规则不支持继续支付 Stage-1 成本。
- 不修改或提交本地忽略的 `docs/opd_proxy_gradient_verify_implementation.md`。

## 验证标准

完成文档后应验证：

1. 所有数值及 SHA 与 canonical artifacts 字节内容一致；
2. gate 推导与 `report.json` 的 `classification` 完全一致；
3. 文档明确 main hypothesis 为 `not_evaluated`；
4. 文档明确 Stage 1 未启动；
5. Markdown 无占位符、临时标记、矛盾或模糊的后续动作；
6. Git diff 只包含预期文档与设计/计划文件。
