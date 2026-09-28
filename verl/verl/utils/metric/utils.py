# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any

import numpy as np


def _flatten_metric_values(values: Any) -> np.ndarray:
    """Flatten possibly ragged worker metric values into a numeric 1-D array."""
    flat: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, np.ndarray):
            flat.extend(np.asarray(value, dtype=float).reshape(-1).tolist())
        else:
            flat.append(value)

    visit(values)
    return np.asarray(flat, dtype=float)


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        values = _flatten_metric_values(val)
        if values.size == 0 or np.isnan(values).all():
            metrics[key] = np.nan
        elif "max" in key:
            metrics[key] = np.nanmax(values)
        elif "min" in key:
            metrics[key] = np.nanmin(values)
        else:
            metrics[key] = np.nanmean(values)
    return metrics
