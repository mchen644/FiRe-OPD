from pathlib import Path

from math_eval.cod_prompt import build_cod_messages


def test_build_cod_messages_loads_chain_of_draft_yaml_and_adapts_boxed_format(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
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
  - question: |
      first question?
    answer: |
      1 + 1 = 2. #### 2
  - question: |
      second question?
    answer: |
      2 + 2 = 4. #### 4
""".lstrip(),
        encoding="utf-8",
    )

    messages = build_cod_messages(
        "target question?",
        cod_repo_dir=cod_repo,
        shot=1,
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Think step by step, but only keep minimum draft for each thinking step, "
                "with 5 words at most.\nReturn the answer at the end of the response within \\boxed{}.\n"
                "Q: first question?\n\nA: 1 + 1 = 2. \\boxed{2}\n\nQ: target question?\nA: "
            ),
        },
    ]


def test_build_cod_messages_uses_all_examples_when_shot_is_negative(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
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

    messages = build_cod_messages("target?", cod_repo_dir=cod_repo, shot=-1)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    user_content = messages[0]["content"]
    assert "Q: one?" in user_content
    assert "Q: two?" in user_content
    assert "####" not in user_content
    assert "\\boxed{1}" in user_content
    assert "\\boxed{2}" in user_content


def test_build_cod_messages_strips_fire_opd_verbose_instruction_from_target_question(tmp_path: Path):
    cod_repo = tmp_path / "chain-of-draft"
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
fewshot: []
""".lstrip(),
        encoding="utf-8",
    )

    messages = build_cod_messages(
        "Hard AIME problem text.\nPlease reason step by step, and put your final answer within \\boxed{}.",
        cod_repo_dir=cod_repo,
        shot=0,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    user_content = messages[0]["content"]
    assert "Hard AIME problem text." in user_content
    assert "Please reason step by step" not in user_content
    assert user_content == (
        "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\n"
        "Return the answer at the end of the response within \\boxed{}.\n"
        "Q: Hard AIME problem text.\nA: "
    )
