from pathlib import Path

import pandas as pd

from math_eval.cod_parquet import convert_dataframe_prompts_to_cod, convert_parquet_file


def _write_cod_config(cod_repo: Path) -> None:
    configs = cod_repo / "configs"
    configs.mkdir(parents=True)
    (configs / "gsm8k_cod.yaml").write_text(
        """
system_prompt: |
  Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.
  Return the answer at the end of the response after a separator ####.
format: |
  Q: {question}
  A: {answer}
fewshot:
  - question: one?
    answer: |
      one draft. #### 1
  - question: two?
    answer: |
      two draft. #### 2
""".lstrip(),
        encoding="utf-8",
    )


def test_convert_dataframe_prompts_to_cod_rewrites_prompt_and_preserves_other_columns(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
    _write_cod_config(cod_repo)
    df = pd.DataFrame(
        [
            {
                "data_source": "DeepMath-103K",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Hard training problem.\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "42"},
                "extra_info": {"index": 7, "split": "train"},
            }
        ]
    )

    converted = convert_dataframe_prompts_to_cod(df, cod_repo_dir=cod_repo, shot=1)

    assert list(converted.columns) == list(df.columns)
    assert converted.loc[0, "data_source"] == "DeepMath-103K"
    assert converted.loc[0, "reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert converted.loc[0, "extra_info"] == {"index": 7, "split": "train"}
    messages = converted.loc[0, "prompt"]
    assert messages == [
        {
            "role": "user",
            "content": (
                "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\n"
                "Return the answer at the end of the response within \\boxed{}.\n"
                "Q: one?\nA: one draft. \\boxed{1}\n\nQ: Hard training problem.\nA: "
            ),
        }
    ]


def test_convert_dataframe_prompts_to_cod_can_preserve_raw_prompt_while_rewriting_student_prompt(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
    _write_cod_config(cod_repo)
    original_prompt = [
        {
            "role": "user",
            "content": "Hard training problem.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    df = pd.DataFrame(
        [
            {
                "data_source": "DeepMath-103K",
                "prompt": original_prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "42"},
                "extra_info": {"index": 7, "split": "train"},
            }
        ]
    )

    converted = convert_dataframe_prompts_to_cod(
        df,
        cod_repo_dir=cod_repo,
        shot=1,
        output_prompt_key="prompt",
        preserve_original_prompt_key="teacher_prompt",
    )

    assert list(converted.columns) == [*df.columns, "teacher_prompt"]
    assert converted.loc[0, "teacher_prompt"] == original_prompt
    assert converted.loc[0, "prompt"] == [
        {
            "role": "user",
            "content": (
                "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\n"
                "Return the answer at the end of the response within \\boxed{}.\n"
                "Q: one?\nA: one draft. \\boxed{1}\n\nQ: Hard training problem.\nA: "
            ),
        }
    ]


def test_convert_dataframe_prompts_to_cod_can_write_teacher_prompt_column(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
    _write_cod_config(cod_repo)
    original_prompt = [
        {
            "role": "user",
            "content": "Hard training problem.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    df = pd.DataFrame(
        [
            {
                "data_source": "DeepMath-103K",
                "prompt": original_prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "42"},
                "extra_info": {"index": 7, "split": "train"},
            }
        ]
    )

    converted = convert_dataframe_prompts_to_cod(
        df,
        cod_repo_dir=cod_repo,
        shot=1,
        output_prompt_key="teacher_prompt",
    )

    assert list(converted.columns) == [*df.columns, "teacher_prompt"]
    assert converted.loc[0, "prompt"] == original_prompt
    messages = converted.loc[0, "teacher_prompt"]
    assert messages == [
        {
            "role": "user",
            "content": (
                "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\n"
                "Return the answer at the end of the response within \\boxed{}.\n"
                "Q: one?\nA: one draft. \\boxed{1}\n\nQ: Hard training problem.\nA: "
            ),
        }
    ]


def test_convert_parquet_file_writes_cod_prompt_parquet(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
    _write_cod_config(cod_repo)
    input_file = tmp_path / "input.parquet"
    output_file = tmp_path / "nested" / "output.parquet"
    pd.DataFrame(
        [
            {
                "id": "AIME2024_0",
                "data_source": "AIME2024",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Eval problem.\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "5"},
                "extra_info": {"index": 0, "split": "test"},
            }
        ]
    ).to_parquet(input_file)

    convert_parquet_file(input_file, output_file, cod_repo_dir=cod_repo, shot=0)

    out = pd.read_parquet(output_file)
    messages = out.loc[0, "prompt"]
    assert len(out) == 1
    assert out.loc[0, "id"] == "AIME2024_0"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Please reason step by step" not in messages[0]["content"]
    assert "Q: Eval problem.\nA: " in messages[0]["content"]
