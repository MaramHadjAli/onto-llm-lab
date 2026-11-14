"""Hand-written individuals for Pizza and Wine. No LLM call.

HermiT flags a fixture when the run is inconsistent, errors, or reports an
unsatisfiable class that the ontology does not already report on its own.
SHACL flags it when ``conforms`` is false. The undefined-term detector flags
an IRI that ``undefined_terms`` returns.

Wine and Food declare ``xml:base`` with a trailing ``#``. The HermiT loader
strips that character from the in-memory stream, so class IRIs stay
``ns#Name`` and match the fixture types. The files on disk are unchanged.
"""

from pathlib import Path

import pytest
from rdflib import Graph

from ollab.core.ontology import load_ontology
from ollab.kgen.generate import undefined_terms
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_owl import _add_individuals, check_consistency, load_local_world
from ollab.kgen.validate_shacl import validate_shacl
from tests.support import ONTOLOGIES

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kgen"
PIZZA = ONTOLOGIES / "pizza.owl"
WINE = ONTOLOGIES / "wine.owl"
WINE_NS = "http://www.w3.org/TR/2003/PR-owl-guide-20031209/wine#"
UNDEFINED = "http://ollab.local/undefined#noSuchProperty"

# Measured against the current shapes and the current HermiT loader.
EXPECTED = {
    ("pizza", "valid"): frozenset(),
    ("pizza", "disjoint"): frozenset({"HermiT", "SHACL"}),
    ("pizza", "functional"): frozenset({"SHACL"}),
    ("pizza", "undefined"): frozenset({"undefined-term"}),
    ("wine", "valid"): frozenset(),
    ("wine", "disjoint"): frozenset({"HermiT", "SHACL"}),
    ("wine", "cardinality"): frozenset({"SHACL"}),
    ("wine", "undefined"): frozenset({"undefined-term"}),
}


def _bundle(path: Path):
    ontology = load_ontology(path)
    shapes = build_shapes(ontology).graph
    baseline = check_consistency(path)
    assert baseline.error is None
    assert baseline.consistent is True
    return ontology, shapes, baseline.unsatisfiable_classes


@pytest.fixture(scope="module")
def pizza():
    return _bundle(PIZZA)


@pytest.fixture(scope="module")
def wine():
    return _bundle(WINE)


def _run(ontology_path: Path, fixture: Path, ontology, shapes, baseline_unsat: tuple[str, ...]):
    text = fixture.read_text(encoding="utf-8")
    data = Graph()
    data.parse(data=text, format="turtle")
    consistency = check_consistency(ontology_path, text)
    conforms, violations = validate_shacl(ontology_path, text, shapes)
    missing = undefined_terms(ontology, data)
    flagged: set[str] = set()
    inconsistent = consistency.consistent is False or consistency.error is not None
    new_unsat = set(consistency.unsatisfiable_classes) != set(baseline_unsat)
    if inconsistent or new_unsat:
        flagged.add("HermiT")
    if not conforms:
        flagged.add("SHACL")
    if missing:
        flagged.add("undefined-term")
    return {
        "hermit_consistent": consistency.consistent,
        "hermit_error": consistency.error,
        "hermit_unsat": consistency.unsatisfiable_classes,
        "baseline_unsat": baseline_unsat,
        "shacl_conforms": conforms,
        "shacl": tuple((row.individual, row.message) for row in violations),
        "undefined": missing,
        "flagged_by": flagged,
    }


@pytest.fixture(scope="module")
def reports(pizza, wine):
    bundles = {"pizza": (PIZZA, pizza), "wine": (WINE, wine)}
    found = {}
    for ontology_name, case in EXPECTED:
        ontology_path, bundle = bundles[ontology_name]
        ontology, shapes, baseline = bundle
        fixture = FIXTURES / f"{ontology_name}_{case}.ttl"
        found[(ontology_name, case)] = _run(ontology_path, fixture, ontology, shapes, baseline)
    return found


def test_each_fixture_is_flagged_by_the_expected_component(reports) -> None:
    for key, expected in EXPECTED.items():
        report = reports[key]
        assert report["flagged_by"] == expected, (key, report)
        assert report["hermit_error"] is None


def test_disjoint_and_count_messages_name_the_broken_constraint(reports) -> None:
    pizza_disjoint = reports[("pizza", "disjoint")]
    assert pizza_disjoint["hermit_consistent"] is False
    assert pizza_disjoint["shacl"]
    assert all("disjoint" in message for _who, message in pizza_disjoint["shacl"])

    pizza_functional = reports[("pizza", "functional")]
    assert pizza_functional["hermit_consistent"] is True
    assert pizza_functional["shacl"]
    assert all("hasSpiciness" in message for _who, message in pizza_functional["shacl"])

    wine_disjoint = reports[("wine", "disjoint")]
    assert wine_disjoint["hermit_consistent"] is False
    assert wine_disjoint["shacl"]
    assert all("disjoint" in message for _who, message in wine_disjoint["shacl"])
    for _who, message in wine_disjoint["shacl"]:
        assert "Less than" not in message
        assert "More than" not in message

    wine_card = reports[("wine", "cardinality")]
    assert wine_card["hermit_consistent"] is True
    assert len(wine_card["shacl"]) == 1
    _who, message = wine_card["shacl"][0]
    assert "madeFromGrape" in message
    assert "More than 1" in message


def test_undefined_iri_is_only_the_invented_predicate(reports) -> None:
    for name in ("pizza", "wine"):
        report = reports[(name, "undefined")]
        assert report["undefined"] == (UNDEFINED,)
        assert report["shacl_conforms"] is True
        assert report["hermit_consistent"] is True


def test_wine_class_iris_keep_a_single_hash_and_valid_types_attach() -> None:
    world = load_local_world(WINE)
    iris = {cls.iri for cls in world.classes() if isinstance(getattr(cls, "iri", None), str)}
    assert f"{WINE_NS}LateHarvest" in iris
    assert f"{WINE_NS[:-1]}##LateHarvest" not in iris
    assert all("##" not in iri for iri in iris)
    _add_individuals(world, (FIXTURES / "wine_valid.ttl").read_text(encoding="utf-8"))
    bottle = "http://ollab.local/fixture/wine/bottle"
    found = world.search(iri=bottle)
    assert len(found) == 1
    assert found[0].iri == bottle
    classes = {
        cls.iri: cls for cls in world.classes() if isinstance(getattr(cls, "iri", None), str)
    }
    assert classes[f"{WINE_NS}Wine"] in found[0].is_a
    valid = check_consistency(WINE, (FIXTURES / "wine_valid.ttl").read_text(encoding="utf-8"))
    disjoint = check_consistency(WINE, (FIXTURES / "wine_disjoint.ttl").read_text(encoding="utf-8"))
    assert valid.error is None
    assert valid.consistent is True
    assert disjoint.error is None
    assert disjoint.consistent is False
