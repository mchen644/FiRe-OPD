from pathlib import Path

import pytest
import torch

from math_eval import validate_vanilla_opd_checkpoint as validation


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    pointer_step: int = 50,
    samples_yielded: int = 51_200,
    sampler_batches: int = 50,
    missing_rank: int | None = None,
    include_log_marker: bool = True,
) -> tuple[Path, Path]:
    root = tmp_path / "checkpoint"
    actor = root / "global_step_50" / "actor"
    hf = actor / "huggingface"
    hf.mkdir(parents=True)
    for rank in range(4):
        if rank == missing_rank:
            continue
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"fixture")
    (hf / "config.json").write_text("{}\n", encoding="utf-8")
    (hf / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "latest_checkpointed_iteration.txt").write_text(
        str(pointer_step), encoding="utf-8"
    )
    torch.save(
        {
            "_snapshot": {
                "_snapshot_step": 50,
                "_main_snapshot": {
                    "_sampler_iter_state": {
                        "samples_yielded": samples_yielded
                    },
                    "_sampler_iter_yielded": sampler_batches,
                },
            },
            "_steps_since_snapshot": 0,
            "_iterator_finished": False,
        },
        root / "global_step_50" / "data.pt",
    )
    log = tmp_path / "run.log"
    marker = (
        "step:50 - training/global_step:50 - actor/pg_loss:0.1\n"
        if include_log_marker
        else "step:49 - training/global_step:49 - actor/pg_loss:0.1\n"
    )
    log.write_text(marker, encoding="utf-8")
    return root, log


def test_validate_checkpoint_accepts_exact_step50_consumption(tmp_path: Path) -> None:
    root, log = _checkpoint_fixture(tmp_path)

    report = validation.validate_checkpoint(
        root,
        log,
        expected_step=50,
        expected_samples=51_200,
        expected_batches=50,
        world_size=4,
    )

    assert report["status"] == "complete"
    assert report["samples_yielded"] == 51_200
    assert report["sampler_batches"] == 50


@pytest.mark.parametrize(
    ("fixture_overrides", "error_match"),
    [
        ({"missing_rank": 3}, "rank 3"),
        ({"samples_yielded": 50_176}, "samples_yielded"),
        ({"sampler_batches": 49}, "sampler batches"),
        ({"pointer_step": 40}, "latest checkpoint"),
        ({"include_log_marker": False}, "step-50 log marker"),
    ],
)
def test_validate_checkpoint_rejects_incomplete_or_wrong_state(
    tmp_path: Path,
    fixture_overrides: dict[str, object],
    error_match: str,
) -> None:
    root, log = _checkpoint_fixture(tmp_path, **fixture_overrides)

    with pytest.raises(ValueError, match=error_match):
        validation.validate_checkpoint(
            root,
            log,
            expected_step=50,
            expected_samples=51_200,
            expected_batches=50,
            world_size=4,
        )
