"""Paths shared by tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini.ttl"
ONTOLOGIES = ROOT / "data" / "ontologies"
