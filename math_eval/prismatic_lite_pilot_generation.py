"""Deterministic sampling and quality primitives for Prismatic-lite."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from math_verify import parse as parse_math_answer
from math_verify import verify as verify_math_answer


class GenerationBackend(Protocol):
    def generate(
        self, prompts: Sequence[object], sampling_parameters: Mapping[str, object]
    ) -> list[list[str]]: ...


def _validated_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    if not rows:
        raise ValueError("rows must be nonempty")
    seen: set[str] = set()
    output: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"row {index} id must be a nonempty string")
        if sample_id in seen:
            raise ValueError(f"duplicate row id {sample_id!r}")
        seen.add(sample_id)
        topic = row.get("topic")
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"row {index} topic must be a nonempty string")
        difficulty = row.get("difficulty")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)):
            raise ValueError(f"row {index} difficulty must be numeric")
        if not math.isfinite(float(difficulty)) or float(difficulty) <= 0:
            raise ValueError(f"row {index} difficulty must be positive and finite")
        output.append(row)
    return output


def stratified_calibration_sample(
    rows: Sequence[Mapping[str, object]], *, size: int, seed: int
) -> list[dict[str, object]]:
    validated = _validated_rows(rows)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    if size > len(validated):
        raise ValueError("size cannot exceed row count")

    cells: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in validated:
        cells[(str(row["topic"]), float(row["difficulty"]))].append(row)
    ordered_cells = sorted(cells)
    ideal = {
        cell: size * len(cells[cell]) / len(validated) for cell in ordered_cells
    }
    quotas = {cell: min(len(cells[cell]), math.floor(ideal[cell])) for cell in ordered_cells}
    remaining = size - sum(quotas.values())
    ranked = sorted(
        ordered_cells,
        key=lambda cell: (-(ideal[cell] - math.floor(ideal[cell])), cell),
    )
    while remaining:
        progressed = False
        for cell in ranked:
            if quotas[cell] < len(cells[cell]):
                quotas[cell] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("cannot allocate exact stratified calibration quota")

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(ordered_cells))
    selected: list[Mapping[str, object]] = []
    for cell, child in zip(ordered_cells, child_sequences, strict=True):
        members = cells[cell]
        count = quotas[cell]
        if count:
            generator = np.random.Generator(np.random.PCG64(child))
            positions = generator.choice(len(members), size=count, replace=False)
            selected.extend(members[int(position)] for position in positions)
    selected.sort(key=lambda row: (int(row.get("source_row_index", 0)), str(row["id"])))
    if len(selected) != size or len({row["id"] for row in selected}) != size:
        raise RuntimeError("stratified sample did not produce exact unique membership")
    return [dict(row) for row in selected]


def difficulty_weighted_fewshots(
    rows: Sequence[Mapping[str, object]], *, requests: int, width: int, seed: int
) -> list[list[str]]:
    validated = _validated_rows(rows)
    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        raise ValueError("requests must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= len(validated):
        raise ValueError("width must be in [1, row_count]")
    weights = np.asarray([float(row["difficulty"]) for row in validated], dtype=np.float64)
    weights /= weights.sum()
    children = np.random.SeedSequence(seed).spawn(requests)
    output: list[list[str]] = []
    for child in children:
        generator = np.random.Generator(np.random.PCG64(child))
        positions = generator.choice(
            len(validated), size=width, replace=False, p=weights
        )
        output.append([str(validated[int(position)]["id"]) for position in positions])
    return output


def problem_prompt(examples: Sequence[Mapping[str, object]]) -> str:
    if len(examples) != 5:
        raise ValueError("problem generation requires exactly five examples")
    prompts: list[str] = []
    for index, example in enumerate(examples, start=1):
        prompt = example.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"example {index} prompt must be nonempty")
        prompts.append(f"Example {index}:\n{prompt.strip()}")
    return (
        "Given five example math problems, create one novel non-multiple-choice "
        "math problem that is similar or harder but is not a paraphrase or a "
        "numeric substitution of any example. Return exactly this format:\n"
        "---\n[[Problem]]\n<new problem>\n---\n\n"
        + "\n\n".join(prompts)
    )


_MULTIPLE_CHOICE_OPTION = re.compile(r"(?<!\w)(?:\([A-E]\)|[A-E]\))(?=\s)")


def is_multiple_choice_problem(problem: str) -> bool:
    if not isinstance(problem, str):
        raise ValueError("problem must be a string")
    return len(_MULTIPLE_CHOICE_OPTION.findall(problem)) >= 2


def solution_messages(problem: str) -> list[dict[str, str]]:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be nonempty")
    return [
        {
            "role": "system",
            "content": "Solve the problem rigorously and put the final answer in \\boxed{}.",
        },
        {"role": "user", "content": problem.strip()},
    ]


def parse_generated_problems(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("generation must be a string")
    output: list[str] = []
    for block in text.split("---"):
        stripped = block.strip()
        if not stripped.startswith("[[Problem]]"):
            continue
        problem = stripped[len("[[Problem]]") :].strip()
        if problem:
            output.append(problem)
    return output


def _normalized_problem_text(problem: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", problem).split())


def candidate_id(request_index: int, ordinal: int, problem: str) -> str:
    if request_index < 0 or ordinal < 0:
        raise ValueError("request index and ordinal must be nonnegative")
    normalized = _normalized_problem_text(problem)
    if not normalized:
        raise ValueError("problem must be nonempty")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"prismatic-qwen3-2k-r{request_index:06d}-p{ordinal:02d}-{digest}"


def _last_boxed_expression(response: str) -> str | None:
    search_end = len(response)
    while search_end > 0:
        marker = response.rfind("\\boxed", 0, search_end)
        if marker < 0:
            return None
        cursor = marker + len("\\boxed")
        while cursor < len(response) and response[cursor].isspace():
            cursor += 1
        if cursor < len(response) and response[cursor] == "{":
            depth = 0
            for end in range(cursor, len(response)):
                character = response[end]
                escaped = end > 0 and response[end - 1] == "\\"
                if character == "{" and not escaped:
                    depth += 1
                elif character == "}" and not escaped:
                    depth -= 1
                    if depth == 0:
                        return response[marker : end + 1]
                    if depth < 0:
                        break
        search_end = marker
    return None


def _parsed_boxed_answer(response: str):
    if not isinstance(response, str):
        return None
    final_box = _last_boxed_expression(response)
    if final_box is None:
        return None
    parsed = parse_math_answer(final_box)
    return parsed if parsed else None


def majority_group(responses: Sequence[str]) -> tuple[int, ...]:
    if len(responses) != 3:
        raise ValueError("majority vote requires exactly three responses")
    parsed = [_parsed_boxed_answer(response) for response in responses]
    parent = list(range(3))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(3):
        if parsed[left] is None:
            continue
        for right in range(left + 1, 3):
            if parsed[right] is not None and bool(
                verify_math_answer(parsed[left], parsed[right])
            ):
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(parsed):
        if value is not None:
            groups[find(index)].append(index)
    eligible = [tuple(group) for group in groups.values() if len(group) >= 2]
    if not eligible:
        return ()
    eligible.sort(key=lambda group: (-len(group), group))
    return eligible[0]


def question_level_completion_rows(
    record: Mapping[str, object],
) -> list[dict[str, object]]:
    question_id = record.get("id")
    prompt = record.get("prompt")
    responses = record.get("responses")
    indices = record.get("majority_indices")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question id must be nonempty")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("question prompt must be nonempty")
    if not isinstance(responses, list) or len(responses) != 3 or not all(
        isinstance(response, str) for response in responses
    ):
        raise ValueError("responses must contain exactly three strings")
    if not isinstance(indices, list) or len(indices) < 2:
        raise ValueError("majority_indices must contain at least two indices")
    if len(set(indices)) != len(indices) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 3
        for index in indices
    ):
        raise ValueError("majority_indices are invalid")
    return [
        {
            "id": f"{question_id}.solution-{index}",
            "question_id": question_id,
            "solution_index": index,
            "prompt": prompt.strip(),
            "completion": responses[index],
        }
        for index in sorted(indices)
    ]


_TOKEN_PATTERN = re.compile(r"\\[a-z]+|\w+", flags=re.IGNORECASE | re.UNICODE)


def normalized_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).lower()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def ten_grams(text: str) -> frozenset[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    if len(tokens) < 10:
        return frozenset()
    return frozenset(tuple(tokens[index : index + 10]) for index in range(len(tokens) - 9))


class NearDuplicateIndex:
    def __init__(self, records: Sequence[tuple[str, str]] = ()) -> None:
        self._texts: dict[str, tuple[str, ...]] = {}
        self._grams: dict[str, frozenset[tuple[str, ...]]] = {}
        self._exact: dict[tuple[str, ...], str] = {}
        self._inverted: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for sample_id, text in records:
            self.add(sample_id, text)

    def add(self, sample_id: str, text: str) -> None:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample id must be nonempty")
        if sample_id in self._texts:
            raise ValueError(f"duplicate indexed id {sample_id!r}")
        tokens = normalized_tokens(text)
        if not tokens:
            raise ValueError("indexed text must contain tokens")
        grams = ten_grams(text)
        self._texts[sample_id] = tokens
        self._grams[sample_id] = grams
        self._exact.setdefault(tokens, sample_id)
        for gram in grams:
            self._inverted[gram].add(sample_id)

    def query(self, text: str) -> tuple[str | None, float]:
        tokens = normalized_tokens(text)
        if tokens in self._exact:
            return self._exact[tokens], 1.0
        grams = ten_grams(text)
        if not grams:
            return None, 0.0
        candidates: set[str] = set()
        for gram in grams:
            candidates.update(self._inverted.get(gram, ()))
        best_id: str | None = None
        best_score = 0.0
        for sample_id in sorted(candidates):
            stored = self._grams[sample_id]
            union = len(grams | stored)
            score = len(grams & stored) / union if union else 0.0
            if score > best_score:
                best_id, best_score = sample_id, score
        return best_id, best_score


def parse_semantic_judgment(text: str) -> bool:
    normalized = text.strip()
    if normalized == "equivalent":
        return True
    if normalized == "not_equivalent":
        return False
    raise ValueError(f"invalid semantic judgment {text!r}")
