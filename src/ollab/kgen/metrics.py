"""Scores defined in ``docs/metrics.md``.

The functions here implement those definitions. They do not add a metric.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from rdflib import Graph, URIRef
from rdflib import Literal as RdfLiteral
from rdflib.namespace import RDF

from ollab.core.ontology import Ontology
from ollab.kgen.generate import _axiom_iris, axioms_for_class, undefined_terms
from ollab.kgen.validate_owl import ConsistencyReport
from ollab.kgen.validate_shacl import ShaclViolation


class GraphMetrics(BaseModel):
    """Metrics for one prefix-completed graph, or nulls when it did not parse.

    ``truncated`` is this completion only. An A2 unit is truncated when any
    round is truncated; that roll-up is computed from the rounds, not here.
    """

    model_config = ConfigDict(frozen=True)

    parse_ok: bool
    truncated: bool | None
    n_undefined_terms: int | None
    n_vocabulary: int | None
    undefined_term_rate: float | None
    n_class_as_individual: int | None
    n_object_triples: int | None
    class_as_individual_rate: float | None
    n_individuals: int
    n_shacl_violations_generated: int | None
    shacl_violations_per_individual: float | None
    n_shacl_violations_other: int | None
    inconsistent: bool | None
    new_unsatisfiable_classes: tuple[str, ...] | None
    new_unsatisfiable: bool | None
    coverage_count: int | None
    n_relevant: int | None
    coverage_rate: float | None
    n_distinct_descriptions: int | None
    duplicate_rate: float | None


def generated_individuals(data: Graph) -> frozenset[str]:
    """URI subjects of an ``rdf:type`` triple."""
    return frozenset(
        str(subject)
        for subject, _predicate, obj in data.triples((None, RDF.type, None))
        if isinstance(subject, URIRef) and isinstance(obj, URIRef)
    )


def vocabulary_iris(data: Graph) -> frozenset[str]:
    """Distinct IRIs in the positions ``undefined_terms`` inspects."""
    fresh = {str(subject) for subject in data.subjects() if isinstance(subject, URIRef)}
    found: set[str] = set()
    for _subject, predicate, obj in data:
        if isinstance(predicate, URIRef):
            found.add(str(predicate))
        if predicate == RDF.type and isinstance(obj, URIRef):
            found.add(str(obj))
        if isinstance(obj, URIRef) and predicate != RDF.type and str(obj) not in fresh:
            found.add(str(obj))
    return frozenset(found)


def relevant_signature(
    ontology: Ontology, target_class: str
) -> tuple[frozenset[str], frozenset[str]]:
    """Relevant classes and properties for coverage. The target class is included."""
    named_classes = set(ontology.named_classes())
    named_properties = set(ontology.object_properties()) | set(ontology.data_properties())
    classes = {target_class}
    properties: set[str] = set()
    for axiom in axioms_for_class(ontology, target_class):
        for iri in _axiom_iris(axiom):
            if iri in named_classes:
                classes.add(iri)
            if iri in named_properties:
                properties.add(iri)
    return frozenset(classes), frozenset(properties)


def score_graph(
    ontology: Ontology,
    target_class: str,
    *,
    parsed: Graph | None,
    parse_ok: bool,
    truncated: bool | None,
    consistency: ConsistencyReport | None,
    violations: tuple[ShaclViolation, ...] | None,
    baseline: ConsistencyReport | None,
) -> GraphMetrics:
    """Score one graph. ``violations`` is null when SHACL did not run."""
    if not parse_ok or parsed is None:
        return GraphMetrics(
            parse_ok=False,
            truncated=truncated,
            n_undefined_terms=None,
            n_vocabulary=None,
            undefined_term_rate=None,
            n_class_as_individual=None,
            n_object_triples=None,
            class_as_individual_rate=None,
            n_individuals=0,
            n_shacl_violations_generated=None,
            shacl_violations_per_individual=None,
            n_shacl_violations_other=None,
            inconsistent=None,
            new_unsatisfiable_classes=None,
            new_unsatisfiable=None,
            coverage_count=None,
            n_relevant=None,
            coverage_rate=None,
            n_distinct_descriptions=None,
            duplicate_rate=None,
        )
    undefined = undefined_terms(ontology, parsed)
    vocabulary = vocabulary_iris(parsed)
    undefined_rate = len(undefined) / len(vocabulary) if vocabulary else None
    n_class, n_object = _class_as_individual(ontology, parsed)
    class_rate = n_class / n_object if n_object else None
    individuals = generated_individuals(parsed)
    generated_hits, other_hits, per_individual = _shacl_split(individuals, violations)
    inconsistent, new_classes, new_flag = _consistency_flags(consistency, baseline)
    classes, properties = relevant_signature(ontology, target_class)
    used = _used_signature(parsed, classes, properties)
    n_relevant = len(classes) + len(properties)
    n_distinct, duplicate = _duplicate_stats(parsed, individuals)
    return GraphMetrics(
        parse_ok=True,
        truncated=truncated,
        n_undefined_terms=len(undefined),
        n_vocabulary=len(vocabulary),
        undefined_term_rate=undefined_rate,
        n_class_as_individual=n_class,
        n_object_triples=n_object,
        class_as_individual_rate=class_rate,
        n_individuals=len(individuals),
        n_shacl_violations_generated=generated_hits,
        shacl_violations_per_individual=per_individual,
        n_shacl_violations_other=other_hits,
        inconsistent=inconsistent,
        new_unsatisfiable_classes=new_classes,
        new_unsatisfiable=new_flag,
        coverage_count=used,
        n_relevant=n_relevant,
        coverage_rate=(used / n_relevant) if n_relevant else None,
        n_distinct_descriptions=n_distinct,
        duplicate_rate=duplicate,
    )


def unit_truncated(flags: tuple[bool | None, ...]) -> bool | None:
    """True when any round hit the token cap. Null when a reason is missing."""
    if any(flag is True for flag in flags):
        return True
    if any(flag is None for flag in flags):
        return None
    return False


def _class_as_individual(ontology: Ontology, data: Graph) -> tuple[int, int]:
    classes = set(ontology.named_classes())
    n_class = 0
    n_object = 0
    for _subject, predicate, obj in data:
        if isinstance(obj, URIRef) and predicate != RDF.type:
            n_object += 1
            if str(obj) in classes:
                n_class += 1
    return n_class, n_object


def _shacl_split(
    individuals: frozenset[str],
    violations: tuple[ShaclViolation, ...] | None,
) -> tuple[int | None, int | None, float | None]:
    if violations is None:
        return None, None, None
    generated_hits = sum(1 for row in violations if row.individual in individuals)
    other_hits = len(violations) - generated_hits
    per_individual = generated_hits / len(individuals) if individuals else None
    return generated_hits, other_hits, per_individual


def _consistency_flags(
    consistency: ConsistencyReport | None,
    baseline: ConsistencyReport | None,
) -> tuple[bool | None, tuple[str, ...] | None, bool | None]:
    if consistency is None or consistency.consistent is None:
        return None, None, None
    if consistency.consistent is False:
        return True, None, None
    inconsistent = False
    baseline_ok = baseline is not None and baseline.consistent is True and baseline.error is None
    if not baseline_ok or baseline is None:
        return inconsistent, None, None
    new_classes = tuple(
        sorted(set(consistency.unsatisfiable_classes) - set(baseline.unsatisfiable_classes))
    )
    return inconsistent, new_classes, bool(new_classes)


def _used_signature(data: Graph, classes: frozenset[str], properties: frozenset[str]) -> int:
    used_classes = {
        str(obj)
        for _subject, _predicate, obj in data.triples((None, RDF.type, None))
        if isinstance(obj, URIRef) and str(obj) in classes
    }
    used_properties = {
        str(predicate)
        for _subject, predicate, _obj in data
        if isinstance(predicate, URIRef) and str(predicate) in properties
    }
    return len(used_classes) + len(used_properties)


def _duplicate_stats(data: Graph, individuals: frozenset[str]) -> tuple[int | None, float | None]:
    if not individuals:
        return None, None
    descriptions = [
        _description(data, iri, individuals, frozenset()) for iri in sorted(individuals)
    ]
    n_distinct = len(set(descriptions))
    n_individuals = len(individuals)
    return n_distinct, (n_individuals - n_distinct) / n_individuals


def _description(
    data: Graph,
    iri: str,
    individuals: frozenset[str],
    visiting: frozenset[str],
) -> tuple[object, ...]:
    if iri in visiting:
        return ("cycle",)
    visiting = visiting | {iri}
    rows: list[tuple[object, ...]] = []
    for predicate, obj in data.predicate_objects(URIRef(iri)):
        if isinstance(obj, URIRef) and str(obj) in individuals:
            value: tuple[object, ...] = (
                "individual",
                _description(data, str(obj), individuals, visiting),
            )
        elif isinstance(obj, URIRef):
            value = ("iri", str(obj))
        elif isinstance(obj, RdfLiteral):
            datatype = "" if obj.datatype is None else str(obj.datatype)
            value = ("literal", str(obj), datatype, obj.language or "")
        else:
            value = ("other", str(obj))
        rows.append((str(predicate), value))
    return tuple(sorted(rows))
