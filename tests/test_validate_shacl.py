"""pyshacl reports the focus individual, the shape, and a message."""

from ollab.core.ontology import load_ontology
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_shacl import validate_shacl
from tests.support import MINI

NS = "http://example.org/mini#"


def _data(body: str) -> str:
    return f"@prefix : <{NS}> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\n{body}\n"


def test_disjoint_types_are_a_violation() -> None:
    ontology = load_ontology(MINI)
    shapes = build_shapes(ontology).graph
    conforms, violations = validate_shacl(
        MINI,
        _data(":bad a :Pizza, :PizzaTopping ."),
        shapes,
    )
    assert conforms is False
    assert any(row.individual == f"{NS}bad" and row.message and row.shape for row in violations)


def test_explicit_type_is_required_for_a_property_subject() -> None:
    shapes = build_shapes(load_ontology(MINI)).graph
    _conforms, violations = validate_shacl(
        MINI,
        _data(":top a :PizzaTopping . :top :hasTopping :top ."),
        shapes,
    )
    assert any(row.individual == f"{NS}top" for row in violations)


def test_well_typed_pizza_has_no_domain_violation() -> None:
    shapes = build_shapes(load_ontology(MINI)).graph
    conforms, violations = validate_shacl(
        MINI,
        _data(":one a :Pizza . :cheese a :CheeseTopping . :one :hasTopping :cheese ."),
        shapes,
    )
    individuals = {row.individual for row in violations}
    assert f"{NS}one" not in individuals or conforms is False
    domain_hits = [
        row
        for row in violations
        if row.individual == f"{NS}one" and "hasTopping" in row.message and "Pizza" in row.message
    ]
    assert domain_hits == []
