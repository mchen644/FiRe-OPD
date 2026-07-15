# OPD Efficacy Pilot Results Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Chinese report at `docs/opd_efficacy_pilot_results.md` that accurately records the completed pilot, its pre-registered `no_go` decision, provenance, limitations, and Stage-1 disposition.

**Architecture:** Treat the immutable pilot artifacts as the sole quantitative source of truth. Write one human-readable Markdown document, then cross-check every decision field, metric, count, path, and SHA against `report.json`, `manifest.json`, and `STAGE_COMPLETE.json`; do not rerun analysis or modify experiment artifacts.

**Tech Stack:** Markdown, Python 3.10 standard library, Git.

## Global Constraints

- Write the final report in Chinese at exactly `docs/opd_efficacy_pilot_results.md`.
- Use only completed canonical artifacts under `data/opd_proxy_gradient_verify/efficacy_pilot/` as quantitative sources.
- Preserve the exact labels `pilot_only`, `no_go`, and `not_evaluated`.
- Keep percentiles to four decimal places, matching canonical `report.json`.
- Do not describe the result as downstream/OOD accuracy or as a failure of the main Stage-1 hypothesis.
- State that cross-seed oracle and target-seed dependence are unavailable.
- State that Stage-1 thresholds were not modified and Stage 1 was not launched.
- Mention the two fail-closed attempts only briefly; do not turn the report into an incident postmortem.
- Do not modify experiment artifacts or the locally ignored `docs/opd_proxy_gradient_verify_implementation.md`.

---

### Task 1: Write and verify the standalone pilot results report

**Files:**
- Create: `docs/opd_efficacy_pilot_results.md`
- Read only: `data/opd_proxy_gradient_verify/efficacy_pilot/report.json`
- Read only: `data/opd_proxy_gradient_verify/efficacy_pilot/manifest.json`
- Read only: `data/opd_proxy_gradient_verify/efficacy_pilot/STAGE_COMPLETE.json`
- Read only: `data/opd_proxy_gradient_verify/efficacy_pilot/capture/target/seed_42/manifest.json`
- Read only: `data/opd_proxy_gradient_verify/efficacy_pilot/capture/proxy/seed_42/manifest.json`

**Interfaces:**
- Consumes: immutable JSON fields and artifact SHA-256 values from the completed pilot.
- Produces: one standalone Markdown report whose assertions are mechanically traceable to those artifacts.

- [ ] **Step 1: Confirm the deliverable does not already exist**

Run:

```bash
cd /home/mchen/FiRe-OPD/.worktrees/opd-proxy-gradient-verify-impl
test ! -e docs/opd_efficacy_pilot_results.md
```

Expected: exit code 0. If the file exists, stop and inspect it rather than overwriting unknown content.

- [ ] **Step 2: Extract the canonical values used by the report**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("data/opd_proxy_gradient_verify/efficacy_pilot")
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
complete = json.loads((root / "STAGE_COMPLETE.json").read_text(encoding="utf-8"))

print(json.dumps(report["classification"], ensure_ascii=False, indent=2, sort_keys=True))
print("counts", report["counts"])
print("manifest_sha256", hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
print("report_sha256", hashlib.sha256((root / "report.json").read_bytes()).hexdigest())
print("complete_sha256", hashlib.sha256((root / "STAGE_COMPLETE.json").read_bytes()).hexdigest())
print("native_rollouts", complete["native_rollouts"])
print("generation_seeds", complete["generation_seeds"])
print("algorithm_contract_sha256", complete["algorithm_contract_sha256"])
print("parent_stage1_manifest_sha256", complete["parent_stage1_manifest_sha256"])
print("target_capture_count", manifest["target_capture_count"])
print("proxy_capture_count", manifest["proxy_capture_count"])
PY
```

Expected values include:

```text
decision: no_go
status: pilot_only
main_hypothesis: not_evaluated
candidate / held_out / selected: 250 / 84 / 56
primary / diagnostic K: 25 / 3
random draws: 10000
native_rollouts: 1
generation_seeds: [42]
target / proxy capture count: 334 / 250
```

- [ ] **Step 3: Create the report with the approved structure**

Create `docs/opd_efficacy_pilot_results.md` with these sections and facts:

```markdown
# Vanilla-OPD Proxy-Gradient Efficacy Pilot 结果报告

## 1. 执行摘要

- Classification status: `pilot_only`
- Operational decision: `no_go`
- Main Stage-1 hypothesis: `not_evaluated`
- Stage 1 未启动并保持暂停。

## 2. 实验目标与边界

说明 pilot 只检验 proxy-gradient selected subset 在冻结 target-gradient space 中相对随机子集的表现；不测试 downstream/OOD accuracy，不建立主假设结论，也不修改 Stage-1 thresholds。

## 3. 冻结实验配置

记录 250 candidates、84 held-out、selected size 56、primary/diagnostic K=25/3、K-means seeds 42/43、10,000 uniform 与 10,000 stratified draws、SeedSequence 2026071402、generation seed 42、native n=1，以及 P_pilot/T_pilot。

## 4. 预注册判定规则

准确写出：两个 seeds 的 G-Vendi 与 coverage 均 >=0.90 且 norm 与 signal 均 >=0.25 才为 go；任一 seed 的 G-Vendi 或 coverage <=0.60 即为 no_go；其余为 borderline。

## 5. 执行与完整性

记录 14/14 work units、target/proxy captures 334/250、四个 target replay shards、四个 proxy replay shards、complete-stage validation，以及 Stage-0/Stage-1 frozen hashes 未变。

## 6. 主要结果

| K-means seed | G-Vendi | Coverage | Gradient norm | OPD signal |
|---|---:|---:|---:|---:|
| 42 | 0.0678 | 0.4591 | 0.0839 | 0.2453 |
| 43 | 0.8534 | 0.7928 | 0.5084 | 0.7568 |

## 7. Gate 判定

说明 seed 42 的 G-Vendi=0.0678 与 coverage=0.4591 均 <=0.60，所以按 inclusive rule 直接触发 no_go；seed 43 不能覆盖这一条件。

## 8. 解释与限制

明确 no_go 只是成本闸门；cross-seed oracle 和 target-seed dependence unavailable；不能推导 proxy-gradient selection 普遍无效。

## 9. 执行说明

简述 FSDP dispatch padding 与 native-n finalizer 两次 fail-closed，注明 source identity 更新后未复用 partial artifacts。

## 10. Artifacts 与 provenance

列出 canonical paths、source commit 020ef80f4a726ec71df496e3db49b223c1f5f568、manifest/report/completion SHA、algorithm-contract SHA 和 frozen parent Stage-1 manifest SHA。

## 11. 后续处置

Stage 1 保持暂停；pilot artifacts 不得满足 Stage-1 resume；若未来要偏离 no_go 继续 Stage 1，必须获得新的明确授权，但不得修改已冻结 contract 或 thresholds。
```

Use complete prose rather than leaving the instructional sentences above in the final document.

- [ ] **Step 4: Mechanically verify the report against canonical artifacts**

Run:

```bash
/home/mchen/miniconda3/envs/gvendi-opd/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("data/opd_proxy_gradient_verify/efficacy_pilot")
doc = Path("docs/opd_efficacy_pilot_results.md").read_text(encoding="utf-8")
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
complete = json.loads((root / "STAGE_COMPLETE.json").read_text(encoding="utf-8"))
classification = report["classification"]

required = [
    classification["status"],
    classification["decision"],
    classification["main_hypothesis"],
    "0.0678", "0.4591", "0.0839", "0.2453",
    "0.8534", "0.7928", "0.5084", "0.7568",
    "020ef80f4a726ec71df496e3db49b223c1f5f568",
    hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    hashlib.sha256((root / "report.json").read_bytes()).hexdigest(),
    hashlib.sha256((root / "STAGE_COMPLETE.json").read_bytes()).hexdigest(),
    complete["algorithm_contract_sha256"],
    complete["parent_stage1_manifest_sha256"],
]
missing = [value for value in required if str(value) not in doc]
assert not missing, missing
assert "downstream/OOD accuracy" in doc
assert "Stage 1" in doc and "未启动" in doc
assert "cross-seed oracle" in doc and "target-seed dependence" in doc
assert "主假设失败" not in doc
print("RESULT_DOCUMENT_CANONICAL_CHECK_OK")
PY
```

Expected: `RESULT_DOCUMENT_CANONICAL_CHECK_OK`.

- [ ] **Step 5: Review Markdown scope and repository diff**

Run:

```bash
! rg -n 'T[B]D|T[O]DO|F[I]XME|X[X]X|待[定]' docs/opd_efficacy_pilot_results.md
git diff --check
git status --short
git diff -- docs/opd_efficacy_pilot_results.md
```

Expected: no placeholder matches, no whitespace errors, and the report contains no claims beyond the approved design.

- [ ] **Step 6: Commit the standalone report**

Run:

```bash
git add docs/opd_efficacy_pilot_results.md
git commit -m "docs(opd): report efficacy pilot result"
```

Expected: one commit containing the standalone result report. The ignored implementation-status document and experiment artifacts must not enter Git.
