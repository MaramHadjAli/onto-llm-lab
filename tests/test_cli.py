"""Command line over the mini ontology."""

import json

from typer.testing import CliRunner

from ollab.cli import app
from tests.support import MINI

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_axioms_limit_and_kind_filter() -> None:
    limited = runner.invoke(app, ["axioms", str(MINI), "--limit", "3"])
    assert limited.exit_code == 0
    assert len(limited.stdout.splitlines()) == 3
    kind = runner.invoke(app, ["axioms", str(MINI), "--kind", "Functional", "--limit", "-1"])
    assert kind.exit_code == 0
    lines = kind.stdout.splitlines()
    assert len(lines) == 3
    assert all(line.startswith("Functional\t") for line in lines)
    bad = runner.invoke(app, ["axioms", str(MINI), "--kind", "NotAnAxiom"])
    assert bad.exit_code != 0


def test_summary_json() -> None:
    result = runner.invoke(app, ["summary", str(MINI)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_axioms"] == 27
    assert payload["n_classes"] == 18
    assert payload["iri"] == "http://example.org/mini"
