"""Ontology loading on the fixture and on the vendored public files."""

from __future__ import annotations

import hashlib

import pytest

from ollab.core.ontology import AxiomKind, OntologyError, load_ontology
from tests.support import MINI, ONTOLOGIES

PIZZA_SHA256 = "0de2cd4df0bc09dfd9403df46ae7aa9d26ec31c32cc9a59c514f459c312a751c"
WINE_SHA256 = "bb1666593f9d7ba2ec9cb06dd4fcc17751ac8d523dc33ce22bcee6beeda778eb"
FOOD_SHA256 = "64af31d99ee62f58b7daa18977bbb27973c67a717d5d6317f42400bd53504184"
NS = "http://example.org/mini#"


def test_mini_header_signature_and_axiom_counts() -> None:
    ontology = load_ontology(MINI)
    assert ontology.iri == "http://example.org/mini"
    assert ontology.version_iri == "http://example.org/mini/1.0"
    assert ontology.imports == ()
    assert len(ontology.named_classes()) == 18
    assert len(ontology.object_properties()) == 5
    assert len(ontology.data_properties()) == 1
    axioms = ontology.axioms()
    assert len(axioms) == 27
    assert len({item.axiom_id for item in axioms}) == 27
    counts = ontology.summary().axiom_counts
    assert counts[AxiomKind.SUBCLASS_OF.value] == 8
    assert counts[AxiomKind.CARDINALITY.value] == 4
    assert counts[AxiomKind.EQUIVALENT_CLASSES.value] == 3
    assert counts[AxiomKind.DISJOINT_WITH.value] == 3
    assert counts[AxiomKind.DOMAIN.value] == 3
    assert counts[AxiomKind.RANGE.value] == 3
    assert counts[AxiomKind.FUNCTIONAL.value] == 3
    unsupported = ontology.unsupported()
    assert [item.keyword for item in unsupported] == ["TransitiveProperty"]


def test_label_preference_is_language_then_text() -> None:
    labels = load_ontology(MINI).labels()
    assert labels[NS + "NamedPizza"] == "Named pizza"
    assert labels[NS + "Pizza"] == "Pizza"


def test_axiom_ids_are_stable_across_parses() -> None:
    first = [item.axiom_id for item in load_ontology(MINI).axioms()]
    second = [item.axiom_id for item in load_ontology(MINI).axioms()]
    assert first == second


def test_missing_file_raises() -> None:
    with pytest.raises(OntologyError, match="not found"):
        load_ontology(MINI.parent / "no-such-file.owl")


def test_vendored_bytes_match_recorded_sha256() -> None:
    digest = {
        "pizza.owl": PIZZA_SHA256,
        "wine.owl": WINE_SHA256,
        "food.owl": FOOD_SHA256,
    }
    licenses = (ONTOLOGIES / "LICENSES.md").read_text(encoding="utf-8")
    for name, expected in digest.items():
        payload = (ONTOLOGIES / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        assert expected in licenses


def test_public_ontologies_parse_without_fetching_imports() -> None:
    wine = load_ontology(ONTOLOGIES / "wine.owl")
    food = load_ontology(ONTOLOGIES / "food.owl")
    pizza = load_ontology(ONTOLOGIES / "pizza.owl")
    wine_base = "http://www.w3.org/TR/2003/PR-owl-guide-20031209/wine"
    food_base = "http://www.w3.org/TR/2003/PR-owl-guide-20031209/food"
    assert f"{food_base}" in wine.imports
    assert f"{wine_base}" in food.imports
    assert f"{wine_base}#Wine" in wine.named_classes()
    assert f"{food_base}#PotableLiquid" in food.named_classes()
    pizza_iri = "https://raw.githubusercontent.com/owlcs/pizza-ontology/refs/heads/master/pizza.owl"
    assert f"{pizza_iri}#Pizza" in pizza.named_classes()
    assert pizza.labels()[f"{pizza_iri}#Pizza"] == "Pizza"
    assert pizza.labels()[f"{pizza_iri}#PineKernels"] == "CoberturaPineKernels"
    for ontology in (wine, food, pizza):
        axioms = ontology.axioms()
        assert axioms
        assert len({item.axiom_id for item in axioms}) == len(axioms)
        again = load_ontology(ontology.path).axioms()
        assert [item.axiom_id for item in again] == [item.axiom_id for item in axioms]
