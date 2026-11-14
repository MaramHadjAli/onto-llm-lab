"""Metric definitions from docs/metrics.md. No LLM call."""

from rdflib import Graph

from ollab.core.ontology import load_ontology
from ollab.kgen.metrics import relevant_signature, score_graph, unit_truncated
from ollab.kgen.validate_owl import ConsistencyReport
from ollab.kgen.validate_shacl import ShaclViolation
from tests.support import MINI

NS = "http://example.org/mini#"


def _graph(text: str) -> Graph:
    graph = Graph()
    graph.parse(data=text, format="turtle")
    return graph


def _score(text: str, **kwargs):
    ontology = load_ontology(MINI)
    parsed = _graph(text)
    return score_graph(
        ontology,
        f"{NS}Pizza",
        parsed=parsed,
        parse_ok=True,
        truncated=False,
        consistency=kwargs.get("consistency"),
        violations=kwargs.get("violations", ()),
        baseline=kwargs.get("baseline"),
    )


def test_undefined_term_rate_uses_checked_positions() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza ; :notAProperty :one .\n"
    metrics = _score(body)
    assert metrics.n_undefined_terms == 1
    assert metrics.n_vocabulary == 3
    assert metrics.undefined_term_rate == 1 / 3
    assert metrics.parse_ok is True


def test_class_as_individual_is_a_non_type_object() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza ; :hasTopping :CheeseTopping .\n"
    metrics = _score(body)
    assert metrics.n_class_as_individual == 1
    assert metrics.n_object_triples == 1
    assert metrics.class_as_individual_rate == 1


def test_shacl_counts_split_generated_individuals_from_other_focus_nodes() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza .\n"
    individual = f"{NS}one"
    violations = (
        ShaclViolation(individual=individual, shape="s", message="on the individual"),
        ShaclViolation(individual=f"{NS}CheeseTopping", shape="s", message="on the object"),
    )
    metrics = _score(body, violations=violations)
    assert metrics.n_individuals == 1
    assert metrics.n_shacl_violations_generated == 1
    assert metrics.shacl_violations_per_individual == 1
    assert metrics.n_shacl_violations_other == 1


def test_new_unsatisfiable_classes_are_the_set_difference() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza .\n"
    baseline = ConsistencyReport(consistent=True, unsatisfiable_classes=(f"{NS}Odd",), error=None)
    current = ConsistencyReport(
        consistent=True,
        unsatisfiable_classes=(f"{NS}Odd", f"{NS}Pizza"),
        error=None,
    )
    metrics = _score(body, consistency=current, baseline=baseline)
    assert metrics.inconsistent is False
    assert metrics.new_unsatisfiable_classes == (f"{NS}Pizza",)
    assert metrics.new_unsatisfiable is True

    inconsistent = ConsistencyReport(consistent=False, unsatisfiable_classes=(), error=None)
    failed = _score(body, consistency=inconsistent, baseline=baseline)
    assert failed.inconsistent is True
    assert failed.new_unsatisfiable_classes is None
    assert failed.new_unsatisfiable is None


def test_coverage_counts_types_and_predicates_in_the_relevant_signature() -> None:
    ontology = load_ontology(MINI)
    target = f"{NS}Pizza"
    classes, properties = relevant_signature(ontology, target)
    assert target in classes
    assert f"{NS}hasTopping" in properties
    body = f"@prefix : <{NS}> .\n:one a :Pizza .\n"
    metrics = _score(body)
    assert metrics.coverage_count == 1
    assert metrics.n_relevant == len(classes) + len(properties)
    used = _score(
        f"@prefix : <{NS}> .\n:one a :Pizza ; :hasTopping :two .\n:two a :CheeseTopping .\n"
    )
    assert used.coverage_count == 3


def test_duplicate_rate_collapses_matching_descriptions() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza .\n:two a :Pizza .\n"
    metrics = _score(body)
    assert metrics.n_individuals == 2
    assert metrics.n_distinct_descriptions == 1
    assert metrics.duplicate_rate == 0.5


def test_parse_failure_leaves_rates_null_and_keeps_truncation() -> None:
    ontology = load_ontology(MINI)
    metrics = score_graph(
        ontology,
        f"{NS}Pizza",
        parsed=None,
        parse_ok=False,
        truncated=True,
        consistency=None,
        violations=None,
        baseline=None,
    )
    assert metrics.parse_ok is False
    assert metrics.truncated is True
    assert metrics.undefined_term_rate is None
    assert metrics.class_as_individual_rate is None
    assert metrics.shacl_violations_per_individual is None
    assert metrics.inconsistent is None
    assert metrics.coverage_count is None
    assert metrics.duplicate_rate is None
    assert metrics.n_individuals == 0


def test_unit_truncation_is_true_when_any_round_hit_the_cap() -> None:
    assert unit_truncated((False, False)) is False
    assert unit_truncated((False, True)) is True
    assert unit_truncated((False, None)) is None
    assert unit_truncated((True, None)) is True
