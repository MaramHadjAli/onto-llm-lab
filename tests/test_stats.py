"""Bootstrap intervals and Cohen's kappa."""

import numpy as np
import pytest

from ollab.core.stats import bootstrap_ci, cohens_kappa


def test_constant_sample_interval_is_the_constant() -> None:
    interval = bootstrap_ci([2.0, 2.0, 2.0], n_resamples=20, seed=0)
    assert interval.estimate == 2.0
    assert interval.low == 2.0
    assert interval.high == 2.0
    assert interval.method == "percentile"
    assert interval.n == 3


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    first = bootstrap_ci(values, n_resamples=200, seed=5)
    second = bootstrap_ci(values, n_resamples=200, seed=5)
    assert first == second
    assert first.low <= first.high
    assert first.estimate == 3.75


def test_bootstrap_rejects_empty_and_bad_confidence() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], confidence=1.0)


def test_cohens_kappa_matches_a_hand_count() -> None:
    # Four items. Agreement on 3, so po = 0.75.
    # Margins are (2, 2) and (3, 1), so pe = 0.5 * 0.75 + 0.5 * 0.25 = 0.5.
    # kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5.
    score = cohens_kappa([0, 0, 1, 1], [0, 0, 1, 0])
    assert score == 0.5


def test_cohens_kappa_perfect_and_single_label() -> None:
    assert cohens_kappa(["yes", "no"], ["yes", "no"]) == 1.0
    assert cohens_kappa(["yes", "yes"], ["yes", "yes"]) == 1.0


def test_cohens_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="length"):
        cohens_kappa([1], [1, 2])
    with pytest.raises(ValueError, match="non-empty"):
        cohens_kappa([], [])


def test_bootstrap_accepts_a_custom_statistic() -> None:
    interval = bootstrap_ci([1.0, 3.0, 5.0], statistic=np.median, n_resamples=30, seed=1)
    assert interval.estimate == 3.0
