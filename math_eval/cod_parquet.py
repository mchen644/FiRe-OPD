"""Convert FiRe-OPD parquet prompts to Chain-of-Draft prompts.

The original OPD math data stores prompts as raw chat messages in the parquet
`prompt` column. By default this utility rewrites that column to the same
single-user CoD prompt format used by `math_eval.cod_prompt`, while preserving
all other columns. It can also write the CoD messages to a separate column (for
example `teacher_prompt`) so student rollout keeps the original prompt while the
teacher/ref log-prob path conditions on CoD, or preserve the original prompt in
another column before rewriting `prompt` so student rollout uses CoD while the
teacher/ref path uses the raw prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from math_eval.cod_prompt import (
        adapt_cod_answer_for_boxed,
        adapt_cod_system_prompt_for_boxed,
        load_chain_of_draft_config,
        strip_fire_opd_verbose_instruction,
    )
except ImportError:
    from cod_prompt import (
        adapt_cod_answer_for_boxed,
        adapt_cod_system_prompt_for_boxed,
        load_chain_of_draft_config,
        strip_fire_opd_verbose_instruction,
    )


def _messages_to_list(messages: Any) -> list[dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return list(messages)


def extract_single_user_question(messages: Any) -> str:
    """Extract the user problem text from a raw-chat prompt column value."""
    message_list = _messages_to_list(messages)
    if len(message_list) != 1:
        raise ValueError(f"Expected exactly one prompt message, got {len(message_list)}")
    message = dict(message_list[0])
    if message.get("role") != "user":
        raise ValueError(f"Expected prompt role 'user', got {message.get('role')!r}")
    return str(message.get("content", ""))


def _build_cod_message_factory(
    cod_repo_dir: str | Path | None = None,
    shot: int | None = 5,
    task: str = "gsm8k",
):
    config = load_chain_of_draft_config(cod_repo_dir=cod_repo_dir, task=task, prompt="cod")
    system_prompt = adapt_cod_system_prompt_for_boxed(str(config["system_prompt"]))
    fmt = str(config["format"])
    fewshot = list(config.get("fewshot", []))
    if shot is not None and shot >= 0:
        fewshot = fewshot[:shot]

    formatted_examples = [
        fmt.format(
            question=example["question"],
            answer=adapt_cod_answer_for_boxed(str(example["answer"])),
        )
        for example in fewshot
    ]
    prefix = system_prompt + "\n"
    if formatted_examples:
        prefix += "\n".join(formatted_examples) + "\n"

    def build(question: str) -> list[dict[str, str]]:
        content = prefix + fmt.format(
            question=strip_fire_opd_verbose_instruction(question),
            answer="",
        ).rstrip("\n")
        return [{"role": "user", "content": content}]

    return build


def convert_dataframe_prompts_to_cod(
    dataframe: pd.DataFrame,
    cod_repo_dir: str | Path | None = None,
    shot: int | None = 5,
    task: str = "gsm8k",
    prompt_key: str = "prompt",
    output_prompt_key: str = "prompt",
    preserve_original_prompt_key: str | None = None,
) -> pd.DataFrame:
    """Return a copy of `dataframe` with CoD messages written to `output_prompt_key`.

    When `output_prompt_key` is the default `prompt`, this preserves the legacy
    behavior and rewrites the student prompt. When it is a different column name
    (for example `teacher_prompt`), the original `prompt` column is left intact
    and the CoD prompt is added for the teacher/ref path. When
    `preserve_original_prompt_key` is set, the original `prompt_key` values are
    copied into that column before writing the CoD prompts to `output_prompt_key`.
    """
    if prompt_key not in dataframe.columns:
        raise ValueError(f"Input dataframe must contain a {prompt_key!r} column")
    if preserve_original_prompt_key is not None and preserve_original_prompt_key == output_prompt_key:
        raise ValueError("preserve_original_prompt_key must differ from output_prompt_key")

    build_cod_messages = _build_cod_message_factory(cod_repo_dir=cod_repo_dir, shot=shot, task=task)
    converted = dataframe.copy(deep=True)
    if preserve_original_prompt_key is not None:
        converted[preserve_original_prompt_key] = list(converted[prompt_key])
    converted[output_prompt_key] = [
        build_cod_messages(extract_single_user_question(messages))
        for messages in converted[prompt_key]
    ]
    return converted


def convert_parquet_file(
    input_file: str | Path,
    output_file: str | Path,
    cod_repo_dir: str | Path | None = None,
    shot: int | None = 5,
    task: str = "gsm8k",
    prompt_key: str = "prompt",
    output_prompt_key: str = "prompt",
    preserve_original_prompt_key: str | None = None,
) -> None:
    """Convert one parquet file's prompt column to CoD and write it out."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    dataframe = pd.read_parquet(input_path)
    converted = convert_dataframe_prompts_to_cod(
        dataframe,
        cod_repo_dir=cod_repo_dir,
        shot=shot,
        task=task,
        prompt_key=prompt_key,
        output_prompt_key=output_prompt_key,
        preserve_original_prompt_key=preserve_original_prompt_key,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output_path, index=False)
    print(f"wrote {len(converted)} rows: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite OPD parquet prompts to Chain-of-Draft prompts.")
    parser.add_argument("--input_file", required=True, help="Input parquet file")
    parser.add_argument("--output_file", required=True, help="Output parquet file")
    parser.add_argument("--cod_repo_dir", default=None, help="Path to the Chain-of-Draft repository")
    parser.add_argument("--cod_task", default="gsm8k", help="Chain-of-Draft config task name")
    parser.add_argument("--cod_shot", "--shot", dest="cod_shot", type=int, default=5)
    parser.add_argument("--prompt_key", default="prompt", help="Input prompt column to read")
    parser.add_argument(
        "--output_prompt_key",
        default="prompt",
        help="Output column for CoD messages; use teacher_prompt to preserve student prompt",
    )
    parser.add_argument(
        "--preserve_original_prompt_key",
        default=None,
        help="Optional column that receives the original prompt before CoD prompts are written",
    )
    args = parser.parse_args()

    convert_parquet_file(
        args.input_file,
        args.output_file,
        cod_repo_dir=args.cod_repo_dir,
        shot=args.cod_shot,
        task=args.cod_task,
        prompt_key=args.prompt_key,
        output_prompt_key=args.output_prompt_key,
        preserve_original_prompt_key=args.preserve_original_prompt_key,
    )


if __name__ == "__main__":
    main()
