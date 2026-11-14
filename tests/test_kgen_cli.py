"""The generation CLI refuses mock writes into results/."""

import json
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


def test_class_pool_keeps_a_prefix_of_the_larger_draw() -> None:
    from ollab.core.ontology import load_ontology
    from ollab.kgen.generate import select_classes
    from ollab.kgen.run import KgenConfig, planned_classes

    ontology = load_ontology(MINI)
    pool = select_classes(ontology, 4, 0)
    kept = planned_classes(ontology, n_classes=2, class_seed=0, class_pool=4)
    assert kept == pool[:2]
    try:
        KgenConfig.model_validate(
            {
                "ontology": str(MINI),
                "n_classes": 4,
                "class_pool": 2,
                "n_individuals": 1,
                "seeds": [0],
                "conditions": ["A0"],
                "provider": "mock",
                "model": "mock-1",
                "output": "out.json",
            }
        )
    except ValueError as exc:
        assert "class_pool" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_recompute_rebuilds_tables_from_mock_completions(tmp_path: Path) -> None:
    from ollab.core.llm import make_client
    from ollab.kgen.run import KgenConfig, execute_suite

    calls = {"n": 0}

    def respond(_request) -> str:
        calls["n"] += 1
        return "ind:1 a :Pizza .\n"

    ontology = tmp_path / "mini.ttl"
    ontology.write_text(MINI.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "run"
    config = KgenConfig(
        ontologies=[ontology],
        n_classes=1,
        n_individuals=1,
        seeds=[0],
        replicates=[0],
        conditions=["A0", "A1", "A2"],
        provider="mock",
        model="mock-1",
        temperature=0,
        max_tokens=64,
        prompt_version="v3",
        output=output,
        check=True,
    )
    client = make_client("mock", cache_dir=tmp_path / "cache", responder=respond)
    documents, summary = execute_suite(config, client=client)
    assert calls["n"] == 2
    archive = output / "raw_completions.jsonl"
    lines = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert [line["condition"] for line in lines] == ["A0", "A1", "A2"]
    assert lines[2]["round"] == 0
    assert lines[2]["cached"] is True
    assert lines[2]["replicate_index"] == 0
    assert lines[2]["ollama_version"] is None
    assert lines[2]["raw_text"] == "ind:1 a :Pizza .\n"
    assert lines[2]["prompt_hash"]
    assert "wall_clock_s" in lines[2]
    before = calls["n"]
    result = runner.invoke(app, ["kgen", "recompute", "--from", str(archive)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls["n"] == before
    rebuilt = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert (
        rebuilt["pooled"]["A0"]["parse_ok"]["estimate"]
        == summary["pooled"]["A0"]["parse_ok"]["estimate"]
    )
    assert (
        rebuilt["pooled"]["A2"]["parse_ok"]["estimate"]
        == summary["pooled"]["A2"]["parse_ok"]["estimate"]
    )
    assert rebuilt["totals"]["uncached_calls"] == summary["totals"]["uncached_calls"]
    assert rebuilt["totals"]["truncated_completions"] == summary["totals"]["truncated_completions"]
    written = json.loads((output / "mini.json").read_text(encoding="utf-8"))
    assert written["records"][0]["metrics"]["parse_ok"] is documents[0].records[0].metrics.parse_ok
    assert written["records"][2]["round0_cached"] is True
