"""SHACL shapes generated from a fixed subset of asserted OWL axioms.

See ``docs/shacl_mapping.md``. OWL domain and range axioms are compiled as
constraints on explicit ``rdf:type`` triples. They are not compiled as
inference rules, and this module does not ask pyshacl to run an OWL reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, SH, XSD

from ollab.core.ontology import Axiom, AxiomKind, Expr, Ontology

SHAPE_NS = "http://ollab.local/shapes/"


@dataclass(frozen=True)
class Untranslated:
    """An axiom this mapping refuses to turn into a shape."""

    axiom_id: str
    kind: str
    reason: str


@dataclass(frozen=True)
class ShapeSet:
    """Shapes produced from one ontology, plus the axioms left aside."""

    graph: Graph
    untranslated: tuple[Untranslated, ...]


def build_shapes(ontology: Ontology) -> ShapeSet:
    """Translate supported axioms into a SHACL graph.

    Axiom order follows ``ontology.axioms()``. Shape IRIs are derived from the
    axiom id, so the graph does not depend on blank-node labels.
    """
    graph = Graph()
    graph.bind("sh", SH)
    skipped: list[Untranslated] = []
    for axiom in ontology.axioms():
        before = len(graph)
        reason = _add_axiom(graph, axiom)
        if reason is not None:
            if len(graph) != before:
                raise RuntimeError(f"shape emitter skipped {axiom.axiom_id} after writing triples")
            skipped.append(Untranslated(axiom.axiom_id, axiom.kind.value, reason))
    return ShapeSet(graph=graph, untranslated=tuple(skipped))


def _add_axiom(graph: Graph, axiom: Axiom) -> str | None:
    if axiom.kind is AxiomKind.DOMAIN and axiom.prop and _is_named(axiom.right):
        _node_shape(
            graph,
            axiom,
            target_predicate=SH.targetSubjectsOf,
            target=URIRef(axiom.prop),
            message=(
                f"Every subject of <{axiom.prop}> must have explicit "
                f"rdf:type <{axiom.right.iri}>."
            ),
            class_iri=axiom.right.iri,
        )
        return None
    if axiom.kind is AxiomKind.RANGE and axiom.prop and _is_named(axiom.right):
        _node_shape(
            graph,
            axiom,
            target_predicate=SH.targetObjectsOf,
            target=URIRef(axiom.prop),
            message=(
                f"Every object of <{axiom.prop}> must have explicit "
                f"rdf:type <{axiom.right.iri}>."
            ),
            class_iri=axiom.right.iri,
        )
        return None
    if axiom.kind is AxiomKind.FUNCTIONAL and axiom.prop:
        if axiom.source == "inverseFunctional":
            _count_on_inverse(graph, axiom, axiom.prop)
        else:
            _count_on_property(
                graph, axiom, SH.targetSubjectsOf, axiom.prop, axiom.prop, max_count=1
            )
        return None
    if axiom.kind is AxiomKind.DISJOINT_WITH:
        return _add_disjoint(graph, axiom)
    if axiom.kind is AxiomKind.CARDINALITY:
        return _add_cardinality(graph, axiom)
    if axiom.kind is AxiomKind.SUBCLASS_OF:
        return "named or complex subclass axioms are not emitted as SHACL constraints"
    if axiom.kind is AxiomKind.EQUIVALENT_CLASSES:
        return "equivalence axioms are not emitted; SHACL cannot state the converse"
    return f"no SHACL mapping for {axiom.kind.value}"


def _add_disjoint(graph: Graph, axiom: Axiom) -> str | None:
    if axiom.source == "disjointWith" and _is_named(axiom.left) and _is_named(axiom.right):
        _not_class(graph, axiom, axiom.left.iri or "", axiom.right.iri or "", suffix="ab")
        _not_class(graph, axiom, axiom.right.iri or "", axiom.left.iri or "", suffix="ba")
        return None
    if axiom.source == "AllDisjointClasses":
        named = [item.iri for item in axiom.members if _is_named(item) and item.iri]
        if len(named) < 2 or len(named) != len(axiom.members):
            return "disjoint group contains a non-named class"
        for index, left in enumerate(named):
            for other in named[index + 1 :]:
                _not_class(graph, axiom, left, other, suffix=f"{index}-{other}")
                _not_class(graph, axiom, other, left, suffix=f"{other}-{index}")
        return None
    return "disjoint axiom is not a pair or group of named classes"


def _add_cardinality(graph: Graph, axiom: Axiom) -> str | None:
    if not _is_named(axiom.left) or axiom.right is None or axiom.right.kind != "card":
        return "cardinality axiom is not a named class restricted by one cardinality"
    card = axiom.right
    if not card.prop or card.n is None or card.cardinality is None:
        return "cardinality restriction is incomplete"
    if card.filler is not None and not _is_named(card.filler):
        return "qualified cardinality filler is not a named class"
    minimum, maximum = _bounds(card.cardinality, card.n)
    filler = card.filler.iri if card.filler is not None and card.filler.iri else None
    _count_on_property(
        graph,
        axiom,
        SH.targetClass,
        axiom.left.iri or "",
        card.prop,
        min_count=minimum,
        max_count=maximum,
        qualifier=filler,
    )
    return None


def _bounds(word: str, count: int) -> tuple[int | None, int | None]:
    if word == "min":
        return count, None
    if word == "max":
        return None, count
    return count, count


def _node_shape(
    graph: Graph,
    axiom: Axiom,
    *,
    target_predicate: URIRef,
    target: URIRef,
    message: str,
    class_iri: str | None,
) -> None:
    shape = URIRef(f"{SHAPE_NS}{axiom.axiom_id}")
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, target_predicate, target))
    if class_iri is not None:
        graph.add((shape, SH["class"], URIRef(class_iri)))
    graph.add((shape, SH.message, Literal(message)))


def _not_class(graph: Graph, axiom: Axiom, target_class: str, banned: str, *, suffix: str) -> None:
    shape = URIRef(f"{SHAPE_NS}{axiom.axiom_id}/{suffix}")
    inner = BNode()
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetClass, URIRef(target_class)))
    graph.add((shape, SH["not"], inner))
    graph.add((inner, SH["class"], URIRef(banned)))
    graph.add(
        (
            shape,
            SH.message,
            Literal(
                f"<{target_class}> is disjoint with <{banned}>; "
                "an individual cannot carry both explicit types."
            ),
        )
    )


def _count_on_inverse(graph: Graph, axiom: Axiom, prop: str) -> None:
    shape = URIRef(f"{SHAPE_NS}{axiom.axiom_id}")
    property_shape = BNode()
    inverse = BNode()
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetObjectsOf, URIRef(prop)))
    graph.add((shape, SH["property"], property_shape))
    graph.add((property_shape, SH.path, inverse))
    graph.add((inverse, SH.inversePath, URIRef(prop)))
    graph.add((property_shape, SH.maxCount, Literal(1, datatype=XSD.integer)))
    graph.add(
        (
            shape,
            SH.message,
            Literal(f"<{prop}> is inverse-functional, so each object has at most one subject."),
        )
    )


def _count_on_property(
    graph: Graph,
    axiom: Axiom,
    target_predicate: URIRef,
    target: str,
    prop: str,
    *,
    min_count: int | None = None,
    max_count: int | None = None,
    qualifier: str | None = None,
) -> None:
    shape = URIRef(f"{SHAPE_NS}{axiom.axiom_id}")
    property_shape = BNode()
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, target_predicate, URIRef(target)))
    graph.add((shape, SH["property"], property_shape))
    graph.add((property_shape, SH.path, URIRef(prop)))
    if qualifier is None:
        if min_count is not None:
            graph.add((property_shape, SH.minCount, Literal(min_count, datatype=XSD.integer)))
        if max_count is not None:
            graph.add((property_shape, SH.maxCount, Literal(max_count, datatype=XSD.integer)))
    else:
        value_shape = BNode()
        if min_count is not None:
            graph.add(
                (property_shape, SH.qualifiedMinCount, Literal(min_count, datatype=XSD.integer))
            )
        if max_count is not None:
            graph.add(
                (property_shape, SH.qualifiedMaxCount, Literal(max_count, datatype=XSD.integer))
            )
        graph.add((property_shape, SH.qualifiedValueShape, value_shape))
        graph.add((value_shape, SH["class"], URIRef(qualifier)))
    graph.add(
        (
            shape,
            SH.message,
            Literal(f"Cardinality constraint on <{prop}> for <{target}>."),
        )
    )


def _is_named(expr: Expr | None) -> bool:
    return expr is not None and expr.kind == "iri" and bool(expr.iri)
