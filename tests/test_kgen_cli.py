"""The generation CLI refuses mock writes into results/."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from ollab.cli import app
from tests.support import MINI, ROOT

runner = CliRunner()


def _config(path: Path, output: Path, *, provider: str = "mock") -> None:
    payload = {
        "ontology": str(MINI),
        "n_classes": 1,
        "n_individuals": 1,
        "class_seed": 0,
        "seeds": [0],
        "conditions": ["A0"],
        "provider": provider,
        "model": "mock-1",
        "temperature": 0,
        "max_tokens": 64,
        "output": str(output),
        "validate": False,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_mock_cannot_write_into_results(tmp_path: Path) -> None:
    config = tmp_path / "mock.yaml"
    target = ROOT / "results" / "kgen_mock_should_not_exist.json"
    _config(config, target)
    result = runner.invoke(app, ["kgen", "run", "--config", str(config)])
    assert result.exit_code == 2
    assert "refusing" in result.stderr
    assert not target.exists()


def test_dry_run_prints_prompt_hash_and_writes_nothing(tmp_path: Path) -> None:
    config = tmp_path / "dry.yaml"
    target = ROOT / "results" / "kgen_mock_dry_run.json"
    _config(config, target)
    result = runner.invoke(app, ["kgen", "run", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert '"provider": "mock"' in result.stdout
    assert '"prompt_hash":' in result.stdout
    assert '"seed": 0' in result.stdout
    assert not target.exists()


def test_mock_may_write_outside_results(tmp_path: Path) -> None:
    config = tmp_path / "outside.yaml"
    target = tmp_path / "out.json"
    _config(config, target)
    result = runner.invoke(app, ["kgen", "run", "--config", str(config)])
    assert result.exit_code == 0, result.stderr
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert '"provider": "mock"' in text
    assert '"model": "mock-1"' in text
    assert '"prompt_hash":' in text
    assert '"seed": 0' in text
