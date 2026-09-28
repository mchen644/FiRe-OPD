# Candidate Selection Fixed-batch Refill Design

Date: 2026-06-26
Repository: `/home/mchen/FiRe-OPD`

## Goal

Fix the `quality_gated_correct_compression` OPD run so physical dropping of no-correct teacher-rejected groups does not produce non-divisible actor-update batches such as `DataProto` size 961 with actor DP chunk 4.

## Root Cause

The current implementation runs candidate selection once per dataloader batch and immediately calls actor update. With `rollout.n=2` and `keep_per_uid=1`, the intended selected size is `data.train_batch_size` (1024 in the failing run). When `drop_rejected_no_correct=True`, the selected size becomes data-dependent. One step selected 960 rows and happened to be divisible by 4; the next selected 961 rows and failed in `DataProto.chunk(chunks=4)`.

DAPO-style filtering avoids this by treating filtering as a sampling stage: keep generating/filtering until a fixed-size training batch is available, then truncate to the configured size before actor update.

## Design

Add a small candidate-selection accumulator that:

1. accepts each already-selected `DataProto` batch,
2. concatenates it with any pending selected rows,
3. returns `None` until at least `target_size = data.train_batch_size * keep_per_uid` rows are available,
4. returns exactly `target_size` rows when ready,
5. discards over-collected rows by truncation, matching the existing DAPO recipe behavior.

Integrate this accumulator in `RayPPOTrainer.fit` immediately after `select_short_correct_candidates()` and before rollout correction, advantage computation, critic update, and actor update.

When the accumulator is not ready, the trainer skips the optimization step, does not increment `global_steps`, and proceeds to the next dataloader batch. Actor weights therefore remain unchanged while multiple selected shards are collected.

When the accumulator becomes ready, recompute `batch.meta_info["global_token_num"]` for the fixed final batch, then continue through rollout correction, advantage computation, and update.

## Metrics

Add per-update accumulator metrics:

- `candidate_selection/accumulated_batches`: number of generation/filter batches used for this update.
- `candidate_selection/accumulated_selected_size`: selected rows available before final truncation.
- `candidate_selection/final_selected_size`: rows sent to update; should equal `data.train_batch_size * keep_per_uid`.
- `candidate_selection/truncated_size`: over-collected rows discarded.

Existing candidate-selection metrics remain available for the most recent generation/filter batch. They are diagnostic, while the new accumulator metrics verify the fixed-size actor-update invariant.

## Testing

Add unit tests for the accumulator helper:

- first selected shard below target returns `None` and stores the shard;
- second shard makes the batch ready;
- final batch is exactly target size;
- over-collected rows are truncated;
- metrics report accumulated batches, pre-truncation size, final size, and truncation count.

Run the existing candidate-selection tests and relevant trainer import/compile checks.
