"""Load an OWL ontology and list the asserted axioms we can verbalise.

``axioms()`` returns asserted RDF axioms, not the OWL deductive closure.
A triple ``C owl:intersectionOf (...)`` on a named class is read as an
``EquivalentClasses`` axiom, which is how the OWL Guide encodes definitions.
Nested descriptions are part of the parent axiom, not extra axioms.
Constructs outside the verbaliser's set are returned by ``unsupported()``
instead of being dropped quietly.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rdflib import BNode, Graph, URIRef
from rdflib import Literal as RdfLiteral
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.term import Node

MAX_EXPR_DEPTH = 64

CardWord = Literal["min", "max", "exactly"]
ExprKind = Literal[
    "iri",
    "literal",
    "some",
    "only",
    "value",
    "card",
    "and",
    "or",
    "not",
    "one_of",
    "unknown",
]
AxiomSource = Literal[
    "subClassOf",
    "equivalentClass",
    "disjointWith",
    "AllDisjointClasses",
    "domain",
    "range",
    "functional",
    "inverseFunctional",
]


class OntologyError(ValueError):
    """The ontology file could not be read as RDF."""


class AxiomKind(StrEnum):
    """Axiom families the verbaliser knows."""

    SUBCLASS_OF = "SubClassOf"
    EQUIVALENT_CLASSES = "EquivalentClasses"
    DISJOINT_WITH = "DisjointWith"
    DOMAIN = "Domain"
    RANGE = "Range"
    CARDINALITY = "Cardinality"
    FUNCTIONAL = "Functional"


class Expr(BaseModel):
    """A class or data-range expression with a stable, id-free structure."""

    model_config = ConfigDict(frozen=True)

    kind: ExprKind
    iri: str | None = None
    prop: str | None = None
    filler: Expr | None = None
    operands: tuple[Expr, ...] = ()
    cardinality: CardWord | None = None
    n: int | None = None
    value: str | None = None
    detail: str | None = None


class Axiom(BaseModel):
    """One asserted axiom. ``axiom_id`` is a hash of the other fields."""

    model_config = ConfigDict(frozen=True)

    kind: AxiomKind
    axiom_id: str
    source: AxiomSource
    left: Expr | None = None
    right: Expr | None = None
    prop: str | None = None
    members: tuple[Expr, ...] = ()


class UnsupportedConstruct(BaseModel):
    """An asserted construct this phase does not verbalise."""

    model_config = ConfigDict(frozen=True)

    keyword: str
    axiom_id: str
    detail: str


class OntologySummary(BaseModel):
    """Counts produced by walking one loaded file."""

    model_config = ConfigDict(frozen=True)

    path: str
    iri: str | None
    version_iri: str | None
    imports: tuple[str, ...]
    n_triples: int
    n_classes: int
    n_object_properties: int
    n_data_properties: int
    n_axioms: int
    axiom_counts: dict[str, int]
    n_unsupported: int
    unsupported_counts: dict[str, int]


@dataclass
class Ontology:
    """An RDF graph loaded from a local file. Imports are not fetched."""

    path: Path
    graph: Graph
    iri: str | None
    version_iri: str | None
    imports: tuple[str, ...]

    def labels(self, lang: str = "en") -> dict[str, str]:
        """Map IRIs to labels, preferring ``lang``, then unlabeled text."""
        return collect_labels(self.graph, lang)

    def named_classes(self) -> tuple[str, ...]:
        """Named ``owl:Class`` IRIs, sorted."""
        return _iris_of_type(self.graph, OWL.Class)

    def object_properties(self) -> tuple[str, ...]:
        """Named ``owl:ObjectProperty`` IRIs, sorted."""
        return _iris_of_type(self.graph, OWL.ObjectProperty)

    def data_properties(self) -> tuple[str, ...]:
        """Named ``owl:DatatypeProperty`` IRIs, sorted."""
        return _iris_of_type(self.graph, OWL.DatatypeProperty)

    def axioms(self) -> tuple[Axiom, ...]:
        """Supported asserted axioms, sorted by kind then id."""
        found = iter_axioms(self.graph)
        return tuple(sorted(found, key=lambda item: (item.kind.value, item.axiom_id)))

    def unsupported(self) -> tuple[UnsupportedConstruct, ...]:
        """Recognised constructs that ``axioms()`` does not return."""
        found = iter_unsupported(self.graph)
        return tuple(sorted(found, key=lambda item: (item.keyword, item.axiom_id)))

    def summary(self) -> OntologySummary:
        """Return triple, signature, and axiom counts for this file."""
        axioms = self.axioms()
        unsupported = self.unsupported()
        return OntologySummary(
            path=str(self.path),
            iri=self.iri,
            version_iri=self.version_iri,
            imports=self.imports,
            n_triples=len(self.graph),
            n_classes=len(self.named_classes()),
            n_object_properties=len(self.object_properties()),
            n_data_properties=len(self.data_properties()),
            n_axioms=len(axioms),
            axiom_counts=dict(sorted(Counter(item.kind.value for item in axioms).items())),
            n_unsupported=len(unsupported),
            unsupported_counts=dict(sorted(Counter(item.keyword for item in unsupported).items())),
        )


def load_ontology(path: str | Path) -> Ontology:
    """Parse a local OWL/RDF file. This function does not follow ``owl:imports``."""
    source = Path(path)
    if not source.is_file():
        raise OntologyError(f"ontology file not found: {source}")
    graph = Graph()
    try:
        graph.parse(source)
    except Exception as exc:  # rdflib raises several parser-specific errors
        raise OntologyError(f"failed to parse {source}: {exc}") from exc
    iri, version_iri, imports = _ontology_header(graph)
    return Ontology(
        path=source.resolve(),
        graph=graph,
        iri=iri,
        version_iri=version_iri,
        imports=imports,
    )


def collect_labels(graph: Graph, lang: str = "en") -> dict[str, str]:
    """Return one label per subject.

    Preference order for a subject is: an exact ``lang`` tag, then a literal
    with no language, then any other language. Ties break by language tag,
    then by the label text, so the choice does not depend on triple order.
    """
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for subject, obj in graph.subject_objects(RDFS.label):
        if not isinstance(subject, URIRef) or not isinstance(obj, RdfLiteral):
            continue
        text = str(obj).strip()
        if not text:
            continue
        language = obj.language or ""
        if language == lang:
            rank = 0
        elif language == "":
            rank = 1
        else:
            rank = 2
        grouped.setdefault(str(subject), []).append((rank, language, text))
    return {iri: sorted(options)[0][2] for iri, options in sorted(grouped.items())}


def iter_axioms(graph: Graph) -> list[Axiom]:
    """Collect supported axioms from asserted triples and OWL 1 definitions."""
    found: list[Axiom] = []
    for subject, obj in graph.subject_objects(RDFS.subClassOf):
        found.append(_subclass_or_cardinality(graph, subject, obj, "subClassOf"))
    for subject, obj in graph.subject_objects(OWL.equivalentClass):
        found.append(_subclass_or_cardinality(graph, subject, obj, "equivalentClass"))
    for subject, obj in graph.subject_objects(OWL.disjointWith):
        found.append(
            _axiom(
                kind=AxiomKind.DISJOINT_WITH,
                source="disjointWith",
                left=parse_expr(graph, subject),
                right=parse_expr(graph, obj),
            )
        )
    for subject in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        members_node = _unique(graph, subject, OWL.members)
        if members_node is None:
            continue
        members = tuple(parse_expr(graph, item) for item in _rdf_list(graph, members_node))
        if len(members) < 2:
            continue
        found.append(
            _axiom(kind=AxiomKind.DISJOINT_WITH, source="AllDisjointClasses", members=members)
        )
    for subject, obj in graph.subject_objects(RDFS.domain):
        if isinstance(subject, URIRef):
            found.append(
                _axiom(
                    kind=AxiomKind.DOMAIN,
                    source="domain",
                    prop=str(subject),
                    right=parse_expr(graph, obj),
                )
            )
    for subject, obj in graph.subject_objects(RDFS.range):
        if isinstance(subject, URIRef):
            found.append(
                _axiom(
                    kind=AxiomKind.RANGE,
                    source="range",
                    prop=str(subject),
                    right=parse_expr(graph, obj),
                )
            )
    for subject in graph.subjects(RDF.type, OWL.FunctionalProperty):
        if isinstance(subject, URIRef):
            found.append(_axiom(kind=AxiomKind.FUNCTIONAL, source="functional", prop=str(subject)))
    for subject in graph.subjects(RDF.type, OWL.InverseFunctionalProperty):
        if isinstance(subject, URIRef):
            found.append(
                _axiom(kind=AxiomKind.FUNCTIONAL, source="inverseFunctional", prop=str(subject))
            )
    found.extend(_defined_class_axioms(graph))
    return found


def iter_unsupported(graph: Graph) -> list[UnsupportedConstruct]:
    """List constructs the verbaliser deliberately does not cover."""
    found: list[UnsupportedConstruct] = []
    type_map = {
        OWL.TransitiveProperty: "TransitiveProperty",
        OWL.SymmetricProperty: "SymmetricProperty",
        OWL.AsymmetricProperty: "AsymmetricProperty",
        OWL.ReflexiveProperty: "ReflexiveProperty",
        OWL.IrreflexiveProperty: "IrreflexiveProperty",
        OWL.AllDifferent: "AllDifferent",
    }
    for type_iri, construct in type_map.items():
        for subject in graph.subjects(RDF.type, type_iri):
            found.append(_unsupported(construct, _signature(graph, subject)))
    predicate_map = {
        OWL.inverseOf: "inverseOf",
        OWL.propertyDisjointWith: "propertyDisjointWith",
        OWL.disjointUnionOf: "disjointUnionOf",
        OWL.propertyChainAxiom: "propertyChainAxiom",
        OWL.hasKey: "hasKey",
        OWL.equivalentProperty: "equivalentProperty",
        OWL.sameAs: "sameAs",
        OWL.differentFrom: "differentFrom",
        RDFS.subPropertyOf: "subPropertyOf",
    }
    for predicate, construct in predicate_map.items():
        for subject, obj in graph.subject_objects(predicate):
            detail = f"{_signature(graph, subject)} {predicate} {_signature(graph, obj)}"
            found.append(_unsupported(construct, detail))
    return found


def parse_expr(graph: Graph, node: Node, depth: int = 0) -> Expr:
    """Parse ``node`` as a class expression. Named IRIs are not expanded."""
    if isinstance(node, URIRef):
        return Expr(kind="iri", iri=str(node))
    if isinstance(node, RdfLiteral):
        return Expr(kind="literal", value=str(node))
    if not isinstance(node, BNode):
        return Expr(kind="unknown", detail=f"term {type(node).__name__}")
    return _parse_complex(graph, node, set(), depth)


def _parse_complex(graph: Graph, node: Node, stack: set[Node], depth: int) -> Expr:
    if depth > MAX_EXPR_DEPTH:
        return Expr(kind="unknown", detail="max-depth")
    if node in stack:
        return Expr(kind="unknown", detail="cycle")
    stack.add(node)
    try:
        is_restriction = _looks_like_restriction(graph, node)
        boolean = _boolean_kind(graph, node)
        if is_restriction and boolean is not None:
            return Expr(kind="unknown", detail="mixed-expression")
        if is_restriction:
            return _parse_restriction(graph, node, depth)
        if boolean == "not":
            filler = _unique(graph, node, OWL.complementOf)
            if filler is None:
                return Expr(kind="unknown", detail="bad-complement")
            return Expr(kind="not", filler=parse_expr(graph, filler, depth + 1))
        if boolean in {"and", "or", "one_of"}:
            predicate = {
                "and": OWL.intersectionOf,
                "or": OWL.unionOf,
                "one_of": OWL.oneOf,
            }[boolean]
            head = _unique(graph, node, predicate)
            if head is None:
                return Expr(kind="unknown", detail=f"bad-{boolean}")
            operands = tuple(parse_expr(graph, item, depth + 1) for item in _rdf_list(graph, head))
            if not operands:
                return Expr(kind="unknown", detail="empty-list")
            return Expr(kind=boolean, operands=operands)
        return Expr(kind="unknown", detail=_signature(graph, node))
    finally:
        stack.remove(node)


def _subclass_or_cardinality(
    graph: Graph,
    subject: Node,
    obj: Node,
    source: Literal["subClassOf", "equivalentClass"],
) -> Axiom:
    left = parse_expr(graph, subject)
    right = parse_expr(graph, obj)
    kind = AxiomKind.CARDINALITY if right.kind == "card" else _kind_for_source(source)
    return _axiom(kind=kind, source=source, left=left, right=right)


def _kind_for_source(source: Literal["subClassOf", "equivalentClass"]) -> AxiomKind:
    if source == "subClassOf":
        return AxiomKind.SUBCLASS_OF
    return AxiomKind.EQUIVALENT_CLASSES


def _defined_class_axioms(graph: Graph) -> list[Axiom]:
    """OWL 1 named-class definitions (``intersectionOf`` on the class itself)."""
    predicates = (OWL.intersectionOf, OWL.unionOf, OWL.complementOf, OWL.oneOf)
    subjects = {
        subject
        for predicate in predicates
        for subject in graph.subjects(predicate, None)
        if isinstance(subject, URIRef)
    }
    found: list[Axiom] = []
    for subject in subjects:
        definition = _parse_complex(graph, subject, set(), 0)
        found.append(
            _axiom(
                kind=AxiomKind.EQUIVALENT_CLASSES,
                source="equivalentClass",
                left=Expr(kind="iri", iri=str(subject)),
                right=definition,
            )
        )
    return found


def _looks_like_restriction(graph: Graph, node: Node) -> bool:
    markers = (
        OWL.onProperty,
        OWL.someValuesFrom,
        OWL.allValuesFrom,
        OWL.hasValue,
        OWL.minCardinality,
        OWL.maxCardinality,
        OWL.cardinality,
        _owl("minQualifiedCardinality"),
        _owl("maxQualifiedCardinality"),
        _owl("qualifiedCardinality"),
    )
    # Consume each object iterator. A generator object is always truthy.
    if any(any(graph.objects(node, marker)) for marker in markers):
        return True
    return OWL.Restriction in set(graph.objects(node, RDF.type))


def _boolean_kind(graph: Graph, node: Node) -> Literal["and", "or", "not", "one_of"] | None:
    if list(graph.objects(node, OWL.intersectionOf)):
        return "and"
    if list(graph.objects(node, OWL.unionOf)):
        return "or"
    if list(graph.objects(node, OWL.complementOf)):
        return "not"
    if list(graph.objects(node, OWL.oneOf)):
        return "one_of"
    return None


def _parse_restriction(graph: Graph, node: Node, depth: int) -> Expr:
    prop = _one(graph, node, OWL.onProperty)
    if prop == "duplicate" or not isinstance(prop, URIRef):
        return Expr(kind="unknown", detail="restriction-without-property")
    some = _one(graph, node, OWL.someValuesFrom)
    only = _one(graph, node, OWL.allValuesFrom)
    value = _one(graph, node, OWL.hasValue)
    if "duplicate" in {some, only, value}:
        return Expr(kind="unknown", detail="restriction-mode")
    card = _cardinality(graph, node)
    modes = [
        name
        for name, present in (("some", some), ("only", only), ("value", value), ("card", card))
        if present not in {None, "missing"}
    ]
    if len(modes) != 1:
        return Expr(kind="unknown", detail="restriction-mode")
    prop_iri = str(prop)
    if modes[0] == "some" and isinstance(some, (URIRef, BNode, RdfLiteral)):
        return Expr(kind="some", prop=prop_iri, filler=parse_expr(graph, some, depth + 1))
    if modes[0] == "only" and isinstance(only, (URIRef, BNode, RdfLiteral)):
        return Expr(kind="only", prop=prop_iri, filler=parse_expr(graph, only, depth + 1))
    if modes[0] == "value" and isinstance(value, (URIRef, BNode, RdfLiteral)):
        if isinstance(value, URIRef):
            return Expr(kind="value", prop=prop_iri, filler=Expr(kind="iri", iri=str(value)))
        return Expr(kind="value", prop=prop_iri, value=str(value))
    if isinstance(card, Expr):
        return card.model_copy(update={"prop": prop_iri})
    return Expr(kind="unknown", detail="restriction")


def _cardinality(graph: Graph, node: Node) -> Expr | None:
    specs: tuple[tuple[URIRef, CardWord, bool], ...] = (
        (OWL.minCardinality, "min", False),
        (OWL.maxCardinality, "max", False),
        (OWL.cardinality, "exactly", False),
        (_owl("minQualifiedCardinality"), "min", True),
        (_owl("maxQualifiedCardinality"), "max", True),
        (_owl("qualifiedCardinality"), "exactly", True),
    )
    hits = [
        (word, qualified, obj)
        for predicate, word, qualified in specs
        for obj in graph.objects(node, predicate)
    ]
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        return Expr(kind="unknown", detail="ambiguous-cardinality")
    word, qualified, raw = hits[0]
    if not isinstance(raw, RdfLiteral):
        return Expr(kind="unknown", detail="bad-cardinality")
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return Expr(kind="unknown", detail="bad-cardinality")
    filler_node = _unique(graph, node, OWL.onClass)
    if filler_node is None:
        filler_node = _unique(graph, node, _owl("onDataRange"))
    if qualified and filler_node is None:
        return Expr(kind="unknown", detail="qualified-without-class")
    if not qualified and filler_node is not None:
        return Expr(kind="unknown", detail="unqualified-with-class")
    filler = parse_expr(graph, filler_node) if filler_node is not None else None
    return Expr(kind="card", cardinality=word, n=number, filler=filler)


def _rdf_list(graph: Graph, node: Node) -> list[Node]:
    items: list[Node] = []
    seen: set[Node] = set()
    current: Node | None = node
    while current is not None and current != RDF.nil:
        if current in seen:
            raise OntologyError("cycle in RDF list")
        seen.add(current)
        firsts = list(graph.objects(current, RDF.first))
        rests = list(graph.objects(current, RDF.rest))
        if len(firsts) != 1 or len(rests) != 1:
            raise OntologyError("malformed RDF list")
        items.append(firsts[0])
        current = rests[0]
        if len(items) > 10_000:
            raise OntologyError("RDF list exceeds 10000 items")
    return items


def _one(graph: Graph, subject: Node, predicate: URIRef) -> Node | Literal["missing", "duplicate"]:
    values = list(graph.objects(subject, predicate))
    if not values:
        return "missing"
    if len(values) > 1:
        return "duplicate"
    return values[0]


def _unique(graph: Graph, subject: Node, predicate: URIRef) -> Node | None:
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        return None
    return values[0]


def _signature(graph: Graph, node: Node, stack: frozenset[BNode] | None = None) -> str:
    """Id-free description of a node, stable across blank-node renamings."""
    if isinstance(node, URIRef):
        return f"<{node}>"
    if isinstance(node, RdfLiteral):
        return node.n3()
    if not isinstance(node, BNode):
        return repr(node)
    active = stack or frozenset()
    if node in active:
        return "cycle"
    parts = [
        f"<{predicate}> {_signature(graph, obj, active | {node})}"
        for predicate, obj in graph.predicate_objects(node)
    ]
    parts.sort()
    return "[" + " | ".join(parts) + "]"


def _axiom(
    *,
    kind: AxiomKind,
    source: AxiomSource,
    left: Expr | None = None,
    right: Expr | None = None,
    prop: str | None = None,
    members: tuple[Expr, ...] = (),
) -> Axiom:
    draft = Axiom(
        kind=kind,
        axiom_id="",
        source=source,
        left=left,
        right=right,
        prop=prop,
        members=members,
    )
    payload = draft.model_dump(mode="json", exclude={"axiom_id"})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return draft.model_copy(update={"axiom_id": digest})


def _unsupported(keyword: str, detail: str) -> UnsupportedConstruct:
    blob = json.dumps(
        {"detail": detail, "keyword": keyword},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return UnsupportedConstruct(keyword=keyword, axiom_id=digest, detail=detail)


def _ontology_header(graph: Graph) -> tuple[str | None, str | None, tuple[str, ...]]:
    typed = list(graph.subjects(RDF.type, OWL.Ontology))
    referenced: set[Node] = set()
    for predicate in (
        OWL.priorVersion,
        OWL.backwardCompatibleWith,
        OWL.incompatibleWith,
        OWL.imports,
    ):
        referenced.update(graph.objects(None, predicate))
    candidates = [node for node in typed if node not in referenced] or list(typed)
    if not candidates:
        return None, None, ()
    node = sorted(candidates, key=lambda item: (not isinstance(item, URIRef), str(item)))[0]
    version_values = [
        str(value) for value in graph.objects(node, OWL.versionIRI) if isinstance(value, URIRef)
    ]
    version_iri = sorted(version_values)[0] if version_values else None
    imports = tuple(
        sorted(
            str(value) for value in graph.objects(node, OWL.imports) if isinstance(value, URIRef)
        )
    )
    iri = str(node) if isinstance(node, URIRef) else None
    return iri, version_iri, imports


def _iris_of_type(graph: Graph, type_iri: URIRef) -> tuple[str, ...]:
    found = {
        str(subject)
        for subject in graph.subjects(RDF.type, type_iri)
        if isinstance(subject, URIRef)
    }
    return tuple(sorted(found))


def _owl(local: str) -> URIRef:
    return URIRef(f"http://www.w3.org/2002/07/owl#{local}")
