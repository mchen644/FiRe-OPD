"""Utilities for reusing Chain-of-Draft prompts in FiRe-OPD math eval.

The source of truth for CoD wording and few-shot examples is the sibling
`chain-of-draft` repository. This module only adapts that config from the
paper's `####` answer separator to FiRe-OPD's math-eval `\boxed{}` format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_COD_REPO_DIR = Path(__file__).resolve().parents[2] / "chain-of-draft"
BOXED_SYSTEM_INSTRUCTION = r"Return the answer at the end of the response within \boxed{}."
_SEPARATOR_INSTRUCTION_RE = re.compile(
    r"\s*Return the answer at the end of the response after a separator ####\.\s*",
    flags=re.IGNORECASE,
)
_FIRE_OPD_VERBOSE_INSTRUCTION_RE = re.compile(
    r"\s*Please reason step by step, and put your final answer within \\boxed\{\}\.\s*$",
    flags=re.IGNORECASE,
)


def load_chain_of_draft_config(
    cod_repo_dir: str | Path | None = None,
    task: str = "gsm8k",
    prompt: str = "cod",
) -> dict[str, Any]:
    """Load a CoD YAML config from the sibling Chain-of-Draft repository."""
    repo_dir = Path(cod_repo_dir) if cod_repo_dir is not None else DEFAULT_COD_REPO_DIR
    config_path = repo_dir / "configs" / f"{task}_{prompt}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Chain-of-Draft config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Chain-of-Draft config: {config_path}")
    return config


def adapt_cod_system_prompt_for_boxed(system_prompt: str) -> str:
    """Convert CoD's `####` final-answer instruction to FiRe-OPD boxed format."""
    adapted = _SEPARATOR_INSTRUCTION_RE.sub(
        lambda _: "\n" + BOXED_SYSTEM_INSTRUCTION,
        system_prompt.strip(),
    )
    if BOXED_SYSTEM_INSTRUCTION not in adapted:
        adapted = adapted.rstrip() + "\n" + BOXED_SYSTEM_INSTRUCTION
    return adapted.strip()


def adapt_cod_answer_for_boxed(answer: str) -> str:
    r"""Convert a CoD few-shot answer from `draft #### final` to `draft \boxed{final}`."""
    answer = str(answer).strip()
    match = re.search(r"####\s*(.*?)\s*$", answer, flags=re.DOTALL)
    if match is None:
        return answer
    draft = answer[: match.start()].rstrip()
    final_answer = match.group(1).strip()
    if not draft:
        return "\\boxed{" + final_answer + "}"
    return draft + " \\boxed{" + final_answer + "}"


def strip_fire_opd_verbose_instruction(question: str) -> str:
    """Remove FiRe-OPD's verbose math instruction before wrapping with CoD."""
    return _FIRE_OPD_VERBOSE_INSTRUCTION_RE.sub("", str(question)).strip()


def _select_fewshot_examples(fewshot: list[dict[str, Any]], shot: int | None) -> list[dict[str, Any]]:
    if shot is None or shot < 0:
        return fewshot
    return fewshot[:shot]


def build_cod_messages(
    question: str,
    cod_repo_dir: str | Path | None = None,
    shot: int | None = 0,
    task: str = "gsm8k",
) -> list[dict[str, str]]:
    """Build a single user message from Chain-of-Draft's CoD YAML config.

    FiRe-OPD's baseline math prompt places the reasoning instruction in the
    user message. The Chain-of-Draft repo also composes its `system_prompt`
    field into a single user payload before sending it to chat APIs, so this
    adapter keeps the same role placement for a clean ablation.

    Args:
        question: Target math problem text.
        cod_repo_dir: Path to the sibling Chain-of-Draft repository. Defaults
            to `../chain-of-draft` relative to this repo.
        shot: Number of CoD few-shot examples to include. `0` means zero-shot;
            a negative value or `None` means use all examples from the CoD YAML.
        task: Chain-of-Draft task config to use. Defaults to `gsm8k`.
    """
    config = load_chain_of_draft_config(cod_repo_dir=cod_repo_dir, task=task, prompt="cod")
    system_prompt = adapt_cod_system_prompt_for_boxed(str(config["system_prompt"]))
    fmt = str(config["format"])
    fewshot_examples = _select_fewshot_examples(list(config.get("fewshot", [])), shot)

    formatted_examples = [
        fmt.format(
            question=example["question"],
            answer=adapt_cod_answer_for_boxed(str(example["answer"])),
        )
        for example in fewshot_examples
    ]
    user_content = system_prompt + "\n"
    if formatted_examples:
        user_content += "\n".join(formatted_examples) + "\n"
    user_content += fmt.format(question=strip_fire_opd_verbose_instruction(question), answer="").rstrip("\n")

    return [{"role": "user", "content": user_content}]
