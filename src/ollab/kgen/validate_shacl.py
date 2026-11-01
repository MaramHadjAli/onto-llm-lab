"""Run pyshacl and report one row per focus node.

Inference is ``none``. SHACL therefore sees only triples that are present in
the data graph and the ontology graph. It does not apply OWL domain or range
as inferred types. That is the strict reading documented in
``docs/shacl_mapping.md``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pyshacl import validate
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, SH

from ollab.core.ontology import load_ontology


class ShaclViolation(BaseModel):
    """One SHACL result, keyed by the focus individual."""

    model_config = ConfigDict(frozen=True)

    individual: str
    shape: str
    message: str


def validate_shacl(
    ontology_path: Path,
    data_turtle: str,
    shapes: Graph,
) -> tuple[bool, tuple[ShaclViolation, ...]]:
    """Validate Turtle individuals against ``shapes``.

    The ontology file is passed as the SHACL ontology graph so class and
    property IRIs are declared. ``inference='none'`` keeps ``rdf:type`` explicit.
    """
    data = Graph()
    data.parse(data=data_turtle, format="turtle")
    ontology = load_ontology(ontology_path)
    conforms, results, _text = validate(
        data,
        shacl_graph=shapes,
        ont_graph=ontology.graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        allow_warnings=True,
    )
    violations = _read_results(results if isinstance(results, Graph) else Graph())
    return bool(conforms), violations


def _read_results(results: Graph) -> tuple[ShaclViolation, ...]:
    rows: list[ShaclViolation] = []
    for result in results.subjects(RDF.type, SH.ValidationResult):
        rows.append(
            ShaclViolation(
                individual=_term(_one(results, result, SH.focusNode)),
                shape=_term(_one(results, result, SH.sourceShape)),
                message=_term(_one(results, result, SH.resultMessage)),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.individual, row.shape, row.message)))


def _one(graph: Graph, subject, predicate):
    values = list(graph.objects(subject, predicate))
    if not values:
        return None
    return values[0]


def _term(node) -> str:
    if isinstance(node, URIRef):
        return str(node)
    if isinstance(node, Literal):
        return str(node)
    if node is None:
        return ""
    return str(node)
