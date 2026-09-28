#!/usr/bin/env python3
"""Probe first high-entropy/forking-token position on math training rollouts.

This follows the definition used in "Beyond the 80/20 Rule": token entropy is
computed from the model's next-token distribution, and forking tokens are the
top-ratio highest-entropy response tokens within a batch. For each rollout we
report the first token position selected by that top-ratio mask.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _to_python_messages(prompt_value: Any) -> list[dict[str, str]]:
    if isinstance(prompt_value, np.ndarray):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, tuple):
        prompt_value = list(prompt_value)
    messages: list[dict[str, str]] = []
    for item in prompt_value:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
        else:
            role = str(getattr(item, "role", "user"))
            content = str(getattr(item, "content", item))
        messages.append({"role": role, "content": content})
    return messages


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_train_prompts(
    *,
    train_file: str | Path,
    tokenizer_path: str,
    num_prompts: int,
    seed: int,
) -> list[dict[str, Any]]:
    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_parquet(train_file)
    rng = random.Random(int(seed))
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[: int(num_prompts)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    rows: list[dict[str, Any]] = []
    for out_idx, row_idx in enumerate(indices):
        row = df.iloc[row_idx]
        messages = _to_python_messages(row["prompt"])
        prompt_text = _apply_chat_template(tokenizer, messages)
        rows.append(
            {
                "probe_index": out_idx,
                "train_index": int(row_idx),
                "data_source": str(row.get("data_source", "")),
                "messages": messages,
                "prompt_text": prompt_text,
                "ground_truth": row.get("reward_model", {}).get("ground_truth") if isinstance(row.get("reward_model"), dict) else None,
            }
        )
    return rows


def generate_rollouts(
    *,
    prompt_rows: Sequence[dict[str, Any]],
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_new_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=int(tensor_parallel_size),
        max_model_len=int(max_model_len),
        max_num_seqs=int(max_num_seqs),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=True,
        seed=int(seed),
    )
    sampling = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        n=1,
        seed=int(seed),
    )
    outputs = llm.generate([row["prompt_text"] for row in prompt_rows], sampling_params=sampling)
    records: list[dict[str, Any]] = []
    for row, output in zip(prompt_rows, outputs):
        gen = output.outputs[0]
        record = dict(row)
        record.update(
            {
                "response_text": gen.text,
                "response_token_ids": list(gen.token_ids),
                "response_token_count": len(gen.token_ids),
                "finish_reason": getattr(gen, "finish_reason", None),
                "stop_reason": getattr(gen, "stop_reason", None),
            }
        )
        records.append(record)
    return records


def _load_hf_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", **kwargs)
    except Exception as exc:
        print(f"flash_attention_2 load failed ({type(exc).__name__}: {exc}); falling back to default attention")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval().to("cuda")
    return tokenizer, model


def _entropy_from_logits(logits, *, chunk_size: int = 128):
    import torch

    values = []
    for chunk in torch.split(logits, int(chunk_size), dim=0):
        x = chunk.float()
        probs = torch.softmax(x, dim=-1)
        ent = torch.logsumexp(x, dim=-1) - torch.sum(probs * x, dim=-1)
        values.append(ent.detach().cpu())
    return torch.cat(values, dim=0).tolist()


def compute_entropies(
    *,
    records: Sequence[dict[str, Any]],
    model_path: str,
    entropy_chunk_size: int,
) -> list[dict[str, Any]]:
    import torch

    tokenizer, model = _load_hf_model(model_path)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        prompt_ids = tokenizer.encode(row["prompt_text"], add_special_tokens=False)
        response_ids = [int(x) for x in row.get("response_token_ids", [])]
        if not response_ids:
            record = dict(row)
            record.update({"prompt_token_count": len(prompt_ids), "token_entropies": [], "entropy_error": "empty_response"})
            out.append(record)
            continue

        input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids, device="cuda")
        start = max(0, len(prompt_ids) - 1)
        end = start + len(response_ids)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[0, start:end, :]
            if logits.shape[0] != len(response_ids):
                raise RuntimeError(f"entropy length mismatch for row {idx}: logits={logits.shape[0]} response={len(response_ids)}")
            entropies = _entropy_from_logits(logits, chunk_size=int(entropy_chunk_size))
        del input_ids, attention_mask, outputs, logits
        torch.cuda.empty_cache()

        record = dict(row)
        record.update(
            {
                "prompt_token_count": len(prompt_ids),
                "token_entropies": entropies,
                "entropy_mean": float(sum(entropies) / len(entropies)),
                "entropy_max": float(max(entropies)),
            }
        )
        out.append(record)
        if (idx + 1) % 8 == 0:
            print(f"computed entropy for {idx + 1}/{len(records)} rollouts")
    return out


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def add_branch_metrics(records: Sequence[dict[str, Any]], *, top_ratio: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[tuple[float, int, int]] = []
    for rec_idx, row in enumerate(records):
        for pos, entropy in enumerate(row.get("token_entropies", []) or []):
            entries.append((float(entropy), rec_idx, pos))

    top_k = max(1, int(math.ceil(len(entries) * float(top_ratio)))) if entries else 0
    top_entries = sorted(entries, key=lambda x: x[0], reverse=True)[:top_k]
    global_selected: dict[int, set[int]] = {}
    for _entropy, rec_idx, pos in top_entries:
        global_selected.setdefault(rec_idx, set()).add(pos)
    global_threshold = float(top_entries[-1][0]) if top_entries else None

    enriched: list[dict[str, Any]] = []
    for rec_idx, row in enumerate(records):
        entropies = [float(x) for x in row.get("token_entropies", []) or []]
        response_len = len(entropies)
        record = dict(row)
        record.pop("token_entropies", None)  # keep output compact; detailed entropies are not needed for summary.
        record["entropy_top_ratio"] = float(top_ratio)

        selected_global = sorted(global_selected.get(rec_idx, set()))
        if selected_global:
            first = int(selected_global[0])
            record["first_global_top_entropy_pos_0based"] = first
            record["first_global_top_entropy_frac"] = float((first + 1) / response_len)
            record["global_top_entropy_token_count"] = len(selected_global)
        else:
            record["first_global_top_entropy_pos_0based"] = None
            record["first_global_top_entropy_frac"] = None
            record["global_top_entropy_token_count"] = 0

        if entropies:
            seq_top_k = max(1, int(math.ceil(response_len * float(top_ratio))))
            seq_selected = sorted(sorted(range(response_len), key=lambda pos: entropies[pos], reverse=True)[:seq_top_k])
            first_seq = int(seq_selected[0])
            record["first_per_rollout_top_entropy_pos_0based"] = first_seq
            record["first_per_rollout_top_entropy_frac"] = float((first_seq + 1) / response_len)
            record["per_rollout_top_entropy_token_count"] = seq_top_k
            record["per_rollout_top_entropy_threshold"] = float(entropies[sorted(range(response_len), key=lambda pos: entropies[pos], reverse=True)[seq_top_k - 1]])
        else:
            record["first_per_rollout_top_entropy_pos_0based"] = None
            record["first_per_rollout_top_entropy_frac"] = None
            record["per_rollout_top_entropy_token_count"] = 0
            record["per_rollout_top_entropy_threshold"] = None
        enriched.append(record)

    global_fracs = [r["first_global_top_entropy_frac"] for r in enriched if r.get("first_global_top_entropy_frac") is not None]
    seq_fracs = [r["first_per_rollout_top_entropy_frac"] for r in enriched if r.get("first_per_rollout_top_entropy_frac") is not None]
    lengths = [int(r.get("response_token_count") or 0) for r in enriched]
    selected_counts = [int(r.get("global_top_entropy_token_count") or 0) for r in enriched]
    summary = {
        "n_rollouts": len(enriched),
        "n_response_tokens": len(entries),
        "top_ratio": float(top_ratio),
        "global_top_k": top_k,
        "global_entropy_threshold": global_threshold,
        "rollouts_with_any_global_top_entropy_token": int(sum(1 for c in selected_counts if c > 0)),
        "rollout_coverage_global_top_entropy": float(sum(1 for c in selected_counts if c > 0) / len(enriched)) if enriched else 0.0,
        "response_length": _quantiles(lengths),
        "first_global_top_entropy_frac": _quantiles(global_fracs),
        "first_per_rollout_top_entropy_frac": _quantiles(seq_fracs),
        "global_top_entropy_tokens_per_rollout": _quantiles(selected_counts),
        "finish_reason_counts": {str(k): int(v) for k, v in sorted(_count(r.get("finish_reason") for r in enriched).items())},
    }
    return enriched, summary


def _count(values: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "entropy", "summarize", "all"], default="all")
    parser.add_argument("--train-file", default="data/g-opd/DeepMath-103K/train_filtered_level6.parquet")
    parser.add_argument("--model-path", default="models/Qwen3-4B")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--entropy-chunk-size", type=int, default=128)
    parser.add_argument("--top-ratio", type=float, default=0.20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(args.model_path).resolve() if not str(args.model_path).startswith("/") else args.model_path)
    tokenizer_path = args.tokenizer_path or model_path

    prompts_path = out_dir / "sampled_prompts.jsonl"
    rollouts_path = out_dir / "rollouts.jsonl"
    entropy_path = out_dir / "rollouts_with_entropy.jsonl"
    branch_path = out_dir / "branch_records.jsonl"
    summary_path = out_dir / "summary.json"

    if args.stage in {"generate", "all"}:
        prompt_rows = sample_train_prompts(
            train_file=args.train_file,
            tokenizer_path=str(tokenizer_path),
            num_prompts=int(args.num_prompts),
            seed=int(args.sample_seed),
        )
        _write_jsonl(prompts_path, prompt_rows)
        print(f"Wrote sampled prompts: {prompts_path}")
        rollout_rows = generate_rollouts(
            prompt_rows=prompt_rows,
            model_path=model_path,
            tensor_parallel_size=int(args.tensor_parallel_size),
            max_model_len=int(args.max_model_len),
            max_new_tokens=int(args.max_new_tokens),
            max_num_seqs=int(args.max_num_seqs),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.generation_seed),
        )
        _write_jsonl(rollouts_path, rollout_rows)
        print(f"Wrote rollouts: {rollouts_path}")
        gc.collect()

    if args.stage in {"entropy", "all"}:
        rollout_rows = _read_jsonl(rollouts_path)
        entropy_rows = compute_entropies(
            records=rollout_rows,
            model_path=model_path,
            entropy_chunk_size=int(args.entropy_chunk_size),
        )
        _write_jsonl(entropy_path, entropy_rows)
        print(f"Wrote entropy records: {entropy_path}")

    if args.stage in {"summarize", "all"}:
        entropy_rows = _read_jsonl(entropy_path)
        branch_rows, summary = add_branch_metrics(entropy_rows, top_ratio=float(args.top_ratio))
        summary.update(
            {
                "train_file": str(args.train_file),
                "model_path": model_path,
                "num_prompts": int(args.num_prompts),
                "sample_seed": int(args.sample_seed),
                "generation_seed": int(args.generation_seed),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "definition": "global: first response token selected by batch-level top-ratio entropy mask, matching Eq.(6); per_rollout: first token among each rollout's own top-ratio entropy tokens",
            }
        )
        _write_jsonl(branch_path, branch_rows)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote branch records: {branch_path}")
        print(f"Wrote summary: {summary_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
