"""HermiT consistency checks through owlready2, using local OWL files only.

``owlready2.onto_path`` is pointed at the ontology directory before loading.
The requested file is loaded with ``only_local=True``. rdflib records each
``owl:imports`` IRI, and this module loads the matching ``{name}.owl`` from
the catalog into the same world. That second step is required for Wine:
its ontology uses ``rdf:about=""``, and owlready2 stores the import on a
different node, so HermiT would not see Food if we relied on owlready2's
import list. No URL is opened.

rdflib parsing elsewhere does not follow ``owl:imports``. HermiT does, through
this catalog.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from owlready2 import OwlReadyInconsistentOntologyError, World, sync_reasoner_hermit
from pydantic import BaseModel, ConfigDict
from rdflib import Graph

from ollab.core.ontology import load_ontology

_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"
_ERROR_LIMIT = 4000


class ConsistencyReport(BaseModel):
    """Result of one HermiT run.

    ``consistent`` is true or false when HermiT finishes. It is null when the
    reasoner process fails. ``error`` is set only in that failure case.
    An inconsistent ontology is a finished run, so ``error`` stays null.
    """

    model_config = ConfigDict(frozen=True)

    consistent: bool | None
    unsatisfiable_classes: tuple[str, ...]
    error: str | None
    reasoner: str = "HermiT"


@contextmanager
def local_catalog(*directories: Path):
    """Temporarily replace ``owlready2.onto_path`` with local directories."""
    import owlready2

    previous = list(owlready2.onto_path)
    owlready2.onto_path.clear()
    for directory in directories:
        resolved = str(directory.resolve())
        if resolved not in owlready2.onto_path:
            owlready2.onto_path.append(resolved)
    try:
        yield
    finally:
        owlready2.onto_path.clear()
        owlready2.onto_path.extend(previous)


def load_local_world(path: Path, *, catalog: Sequence[Path] | None = None) -> World:
    """Load ``path`` and its local imports into a fresh world.

    The catalog defaults to the file's parent directory, which is where
    ``food.owl`` sits next to ``wine.owl``. Import IRIs are taken from the
    rdflib reading of each file and resolved to ``{name}.owl`` in the catalog.
    Owlready2 does not attach Wine's ``owl:imports`` to the ontology node
    (``rdf:about=""``), so this function loads those files itself.
    """
    source = path.resolve()
    directories = list(catalog) if catalog is not None else [source.parent]
    world = World()
    with local_catalog(*directories):
        _load_file(world, source, directories, set())
    return world


def check_consistency(
    path: Path,
    individuals_turtle: str | None = None,
    *,
    catalog: Sequence[Path] | None = None,
) -> ConsistencyReport:
    """Run HermiT on ``path`` plus optional Turtle individuals.

    Individuals are parsed with rdflib and added as RDF/XML in the same world.
    A Turtle syntax error is reported in ``error`` and does not start HermiT.
    """
    source = path.resolve()
    directories = list(catalog) if catalog is not None else [source.parent]
    world = World()
    try:
        with local_catalog(*directories):
            _load_file(world, source, directories, set())
            if individuals_turtle is not None:
                _add_individuals(world, individuals_turtle)
            sync_reasoner_hermit(world, debug=0)
    except OwlReadyInconsistentOntologyError:
        return ConsistencyReport(consistent=False, unsatisfiable_classes=(), error=None)
    except Exception as exc:
        return _failed(exc)
    return ConsistencyReport(
        consistent=True,
        unsatisfiable_classes=_unsatisfiable_classes(world),
        error=None,
    )


def _load_file(world: World, source: Path, directories: Sequence[Path], seen: set[Path]) -> None:
    """Load one file, then every ``owl:imports`` target found on disk."""
    resolved = source.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    header = load_ontology(resolved)
    iri = header.iri or f"http://ollab.local/file/{resolved.stem}"
    world.get_ontology(iri).load(fileobj=_owlready_stream(resolved), only_local=True)
    for imported in header.imports:
        _load_file(world, _find_import(imported, directories), directories, seen)


def _owlready_stream(source: Path) -> io.BytesIO:
    """Return RDF/XML bytes. Turtle fixtures are converted; OWL/RDF files are not rewritten."""
    if source.suffix.lower() in {".ttl", ".turtle"}:
        graph = Graph()
        graph.parse(source, format="turtle")
        payload = graph.serialize(format="xml")
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        return io.BytesIO(raw)
    return io.BytesIO(source.read_bytes())


def _find_import(iri: str, directories: Sequence[Path]) -> Path:
    name = iri.rstrip("#/").rsplit("/", 1)[-1]
    for directory in directories:
        for candidate in (
            directory / f"{name}.owl",
            directory / f"{name}.rdf",
            directory / f"{name}.ttl",
            directory / name,
        ):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"no local ontology file for import {iri}")


def _add_individuals(world: World, turtle: str) -> None:
    graph = Graph()
    try:
        graph.parse(data=turtle, format="turtle")
    except Exception as exc:
        raise ValueError(f"individual Turtle did not parse: {exc}") from exc
    payload = graph.serialize(format="xml")
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    data_onto = world.get_ontology("http://ollab.local/individuals")
    data_onto.load(fileobj=io.BytesIO(raw), only_local=True)


def _unsatisfiable_classes(world: World) -> tuple[str, ...]:
    from owlready2 import Nothing

    found: list[str] = []
    for cls in world.classes():
        iri = getattr(cls, "iri", None)
        if not isinstance(iri, str) or iri == _NOTHING:
            continue
        equivalents = list(getattr(cls, "equivalent_to", ()))
        if any(item == Nothing or getattr(item, "iri", None) == _NOTHING for item in equivalents):
            found.append(iri)
    return tuple(sorted(set(found)))


def _failed(exc: Exception) -> ConsistencyReport:
    message = f"{type(exc).__name__}: {exc}"
    if len(message) > _ERROR_LIMIT:
        message = message[:_ERROR_LIMIT] + " [truncated]"
    return ConsistencyReport(consistent=None, unsatisfiable_classes=(), error=message)
