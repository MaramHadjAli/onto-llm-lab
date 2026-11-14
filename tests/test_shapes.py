"""SHACL shapes for the mini ontology follow the documented mapping."""

from rdflib.namespace import SH

from ollab.core.ontology import AxiomKind, load_ontology
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


def test_each_disjoint_pair_emits_one_shape() -> None:
    ontology = load_ontology(MINI)
    pairs: set[tuple[str, str]] = set()
    for axiom in ontology.axioms():
        if axiom.kind is not AxiomKind.DISJOINT_WITH:
            continue
        if (
            axiom.source == "disjointWith"
            and axiom.left is not None
            and axiom.right is not None
            and axiom.left.iri
            and axiom.right.iri
        ):
            pairs.add(tuple(sorted((axiom.left.iri, axiom.right.iri))))
        if axiom.source == "AllDisjointClasses":
            named = [item.iri for item in axiom.members if item.iri]
            pairs.update(
                tuple(sorted((left, right)))
                for index, left in enumerate(named)
                for right in named[index + 1 :]
            )
    graph = build_shapes(ontology).graph
    emitted = {
        tuple(sorted((str(graph.value(shape, SH.targetClass)), str(banned))))
        for shape in graph.subjects(SH["not"], None)
        for banned in graph.objects(graph.value(shape, SH["not"]), SH["class"])
    }
    assert emitted == pairs
    assert len(list(graph.subjects(SH["not"], None))) == len(pairs)


def test_cardinality_shape_uses_min_count() -> None:
    graph = build_shapes(load_ontology(MINI)).graph
    counts = {int(obj) for obj in graph.objects(None, SH.minCount)}
    assert 3 in counts
