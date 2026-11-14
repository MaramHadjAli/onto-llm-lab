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
    assert '"prompt_version": "v2"' in result.stdout
    assert '"seed": 0' in result.stdout
    assert not target.exists()


def test_a2_config_requires_a1_before_it_and_validation() -> None:
    from ollab.kgen.run import KgenConfig

    payload = {
        "ontology": str(MINI),
        "n_classes": 1,
        "n_individuals": 1,
        "seeds": [0],
        "conditions": ["A2", "A1"],
        "provider": "mock",
        "model": "mock-1",
        "output": "out.json",
        "validate": True,
    }
    try:
        KgenConfig.model_validate(payload)
    except ValueError as exc:
        assert "A1 must appear before A2" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    payload["conditions"] = ["A1", "A2"]
    payload["validate"] = False
    try:
        KgenConfig.model_validate(payload)
    except ValueError as exc:
        assert "validate: true" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_suite_writes_one_file_per_ontology_and_a_summary(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha.ttl"
    beta = tmp_path / "beta.ttl"
    alpha.write_text(MINI.read_text(encoding="utf-8"), encoding="utf-8")
    beta.write_text(MINI.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "suite"
    config = tmp_path / "suite.yaml"
    payload = {
        "ontologies": [str(alpha), str(beta)],
        "n_classes": 1,
        "n_individuals": 1,
        "class_seed": 0,
        "seeds": [0],
        "conditions": ["A0"],
        "provider": "mock",
        "model": "mock-1",
        "temperature": 0.7,
        "max_tokens": 64,
        "prompt_version": "v3",
        "output": str(output),
        "validate": False,
    }
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = runner.invoke(app, ["kgen", "run", "--config", str(config)])
    assert result.exit_code == 0, result.stderr
    assert (output / "alpha.json").is_file()
    assert (output / "beta.json").is_file()
    summary = (output / "summary.json").read_text(encoding="utf-8")
    assert '"prompt_version": "v3"' in summary
    assert '"n_units": 2' in summary
    assert '"temperature": 0.7' in summary


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
