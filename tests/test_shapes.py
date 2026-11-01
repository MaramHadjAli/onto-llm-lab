"""SHACL shapes for the mini ontology follow the documented mapping."""

from rdflib.namespace import SH

from ollab.core.ontology import load_ontology
from ollab.kgen.shapes import SHAPE_NS, build_shapes
from tests.support import MINI

NS = "http://example.org/mini#"


def test_domain_range_functional_and_disjoint_shapes_exist() -> None:
    shapes = build_shapes(load_ontology(MINI))
    graph = shapes.graph
    targets = {str(obj) for obj in graph.objects(None, SH.targetSubjectsOf)}
    object_targets = {str(obj) for obj in graph.objects(None, SH.targetObjectsOf)}
    classes = {str(obj) for obj in graph.objects(None, SH.targetClass)}
    assert f"{NS}hasTopping" in targets
    assert f"{NS}hasTopping" in object_targets
    assert f"{NS}hasBase" in targets
    assert f"{NS}Pizza" in classes
    assert f"{NS}InterestingPizza" in classes
    assert any(str(subject).startswith(SHAPE_NS) for subject in graph.subjects())
    reasons = {item.reason for item in shapes.untranslated}
    assert any("subclass" in reason for reason in reasons)
    assert any("equivalence" in reason for reason in reasons)


def test_cardinality_shape_uses_min_count() -> None:
    graph = build_shapes(load_ontology(MINI)).graph
    counts = {int(obj) for obj in graph.objects(None, SH.minCount)}
    assert 3 in counts
