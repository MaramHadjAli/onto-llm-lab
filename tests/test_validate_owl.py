"""HermiT reports and offline Wine import resolution."""

import socket
import urllib.request

import owlready2.namespace as owl_namespace
import pytest

from ollab.kgen.validate_owl import check_consistency, load_local_world
from tests.support import MINI, ONTOLOGIES


def test_inconsistent_mini_is_reported_without_an_error() -> None:
    report = check_consistency(MINI.parent / "inconsistent.ttl")
    assert report.consistent is False
    assert report.error is None
    assert report.reasoner == "HermiT"
    assert report.unsatisfiable_classes == ()


def test_unsatisfiable_class_is_listed_when_the_ontology_is_consistent() -> None:
    report = check_consistency(MINI.parent / "unsatisfiable.ttl")
    assert report.error is None
    assert report.consistent is True
    assert "http://example.org/unsat#C" in report.unsatisfiable_classes


def test_added_individual_can_make_a_consistent_ontology_inconsistent() -> None:
    turtle = """
    @prefix : <http://example.org/mini#> .
    :bad a :Pizza, :PizzaTopping .
    """
    report = check_consistency(MINI, turtle)
    assert report.error is None
    assert report.consistent is False


def test_wine_load_does_not_touch_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"network access while loading Wine: {args!r} {kwargs!r}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(owl_namespace.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    world = load_local_world(ONTOLOGIES / "wine.owl")
    ontologies = list(world.ontologies.values())
    iris = {ontology.base_iri.rstrip("#/") for ontology in ontologies if ontology.base_iri}
    assert "http://www.w3.org/TR/2003/PR-owl-guide-20031209/wine" in iris
    assert "http://www.w3.org/TR/2003/PR-owl-guide-20031209/food" in iris
