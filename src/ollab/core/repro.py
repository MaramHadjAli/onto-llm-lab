"""Seeds and run metadata for reproducible experiment records.

``set_seeds`` sets the ``random`` and legacy NumPy global seeds and returns
the ``numpy.random.Generator`` that experiment code should draw from.
``git_head`` is ``None`` when git is missing or ``root`` is not a repository.
The commit is read, never invented.
"""

from __future__ import annotations

import importlib.metadata
import platform
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_COMMIT = re.compile(r"[0-9a-f]{40}")


class RunMetadata(BaseModel):
    """Facts recorded at the start of a run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    seed: int
    created_utc: str
    python: str
    platform: str
    package_version: str
    git_commit: str | None
    command: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def set_seeds(seed: int) -> np.random.Generator:
    """Seed the standard library and NumPy, and return a Generator."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def git_head(root: Path) -> str | None:
    """Return the current commit, or ``None`` if it cannot be read."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip().lower()
    if _COMMIT.fullmatch(commit):
        return commit
    return None


def package_version() -> str:
    """Return the installed distribution version, or a dev marker."""
    try:
        return importlib.metadata.version("onto-llm-lab")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0-dev"


def collect_run_metadata(
    seed: int,
    *,
    root: Path,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
    created_utc: str | None = None,
) -> RunMetadata:
    """Collect environment facts for ``seed``. Pass ``created_utc`` to fix the clock."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    timestamp = created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return RunMetadata(
        run_id=f"seed-{seed}-{timestamp}",
        seed=seed,
        created_utc=timestamp,
        python=sys.version.split()[0],
        platform=platform.platform(),
        package_version=package_version(),
        git_commit=git_head(root),
        command=command,
        extra=dict(extra or {}),
    )


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    """Write ``metadata`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_run_metadata(path: Path) -> RunMetadata:
    """Read metadata written by ``write_run_metadata``."""
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))
