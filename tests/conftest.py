"""Shared paths for the offline test suite."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini.ttl"
ONTOLOGIES = ROOT / "data" / "ontologies"
