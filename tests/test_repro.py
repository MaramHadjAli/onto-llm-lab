"""Seed control and run metadata."""

from pathlib import Path

import numpy as np

from ollab import __version__
from ollab.core.repro import (
    collect_run_metadata,
    git_head,
    package_version,
    read_run_metadata,
    set_seeds,
    write_run_metadata,
)


def test_set_seeds_matches_numpy_generator() -> None:
    generator = set_seeds(0)
    expected = np.random.default_rng(0)
    assert generator.integers(0, 1_000_000) == expected.integers(0, 1_000_000)


def test_metadata_round_trip(tmp_path: Path) -> None:
    metadata = collect_run_metadata(
        3,
        root=tmp_path,
        command="pytest",
        extra={"phase": "0"},
        created_utc="2026-11-01T00:00:00Z",
    )
    assert metadata.git_commit is None
    assert metadata.seed == 3
    assert metadata.created_utc == "2026-11-01T00:00:00Z"
    assert metadata.package_version == package_version()
    path = tmp_path / "run.json"
    write_run_metadata(path, metadata)
    assert read_run_metadata(path) == metadata


def test_git_head_on_a_non_repo_is_none(tmp_path: Path) -> None:
    assert git_head(tmp_path) is None


def test_package_version_matches_the_module() -> None:
    assert __version__ == "0.1.0"
    assert package_version() == "0.1.0"
