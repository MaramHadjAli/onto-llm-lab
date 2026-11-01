"""Command line for ontology loading and Track A generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ollab import __version__
from ollab.core.ontology import AxiomKind, load_ontology
from ollab.core.verbalize import verbalize

app = typer.Typer(help="Ontology-controlled LLM experiments.", no_args_is_help=True)
kgen_app = typer.Typer(help="Track A: ontology-guided generation.")
app.add_typer(kgen_app, name="kgen")


def main() -> None:
    """Console entry point."""
    app()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def axioms(
    path: Annotated[Path, typer.Argument(help="Local OWL or RDF file.")],
    limit: Annotated[int, typer.Option(help="Maximum number of axioms to print.")] = 20,
    kind: Annotated[str | None, typer.Option(help="Keep only this axiom kind.")] = None,
) -> None:
    """Print verbalised axioms from a local ontology file."""
    ontology = load_ontology(path)
    labels = ontology.labels()
    rows = ontology.axioms()
    if kind is not None:
        allowed = {item.value for item in AxiomKind}
        if kind not in allowed:
            names = ", ".join(sorted(allowed))
            raise typer.BadParameter(f"kind must be one of: {names}")
        rows = tuple(row for row in rows if row.kind.value == kind)
    shown = rows if limit < 0 else rows[:limit]
    for row in shown:
        typer.echo(f"{row.kind.value}\t{row.axiom_id}\t{verbalize(row, labels)}")


@app.command()
def summary(
    path: Annotated[Path, typer.Argument(help="Local OWL or RDF file.")],
) -> None:
    """Print axiom and signature counts as JSON."""
    ontology = load_ontology(path)
    typer.echo(json.dumps(ontology.summary().model_dump(mode="json"), indent=2, sort_keys=True))


@kgen_app.command("run")
def kgen_run(
    config: Annotated[Path, typer.Option("--config", help="YAML experiment config.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print JSON and do not write a file.")
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the model name in the config.")
    ] = None,
) -> None:
    """Generate A0/A1 individuals. Mock output cannot be written under results/."""
    from ollab.kgen.run import SafetyError, dump_document, execute, load_config, write_output

    loaded = load_config(config)
    if model is not None:
        loaded = loaded.model_copy(update={"model": model})
    try:
        document = execute(loaded, dry_run=dry_run)
    except SafetyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    text = dump_document(document)
    if dry_run:
        typer.echo(text)
        return
    destination = write_output(loaded.output, document)
    typer.echo(f"wrote {destination}")
