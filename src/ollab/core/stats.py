"""Small statistics used by later experiment reports.

The bootstrap is a percentile interval. Resamples are drawn with replacement
from a 1-D sample using ``numpy.random.Generator(seed)``. The interval is
``numpy.quantile`` at the two-sided tails (default linear interpolation).
Cohen's kappa is unweighted and computed from the confusion matrix of the
two rating sequences. No external table of critical values is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate and percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    n: int
    n_resamples: int
    seed: int
    method: str = "percentile"


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 9999,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Return a percentile bootstrap confidence interval for ``statistic``.

    Paired comparisons can pass the per-item differences as ``values``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    data = np.asarray(list(values), dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a non-empty 1-D sample")
    if not np.isfinite(data).all():
        raise ValueError("values must be finite")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, data.size, size=(n_resamples, data.size))
    resampled = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        resampled[index] = float(statistic(data[draw[index]]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [alpha, 1.0 - alpha])
    return BootstrapCI(
        estimate=float(statistic(data)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n=int(data.size),
        n_resamples=n_resamples,
        seed=seed,
        method="percentile",
    )


def cohens_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> float:
    """Unweighted Cohen's kappa for two categorical rating sequences.

    When both raters use a single identical label, agreement and chance
    agreement are both 1. This function returns 1.0 in that case.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must have the same length")
    count = len(rater_a)
    if count == 0:
        raise ValueError("ratings must be non-empty")
    labels = sorted({*rater_a, *rater_b}, key=lambda item: (str(type(item)), str(item)))
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for left, right in zip(rater_a, rater_b, strict=True):
        matrix[index[left], index[right]] += 1.0
    observed = float(np.trace(matrix) / count)
    expected = float(np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / (count * count))
    if math.isclose(expected, 1.0):
        return 1.0
    return float((observed - expected) / (1.0 - expected))
