"""Template verbalisation of supported OWL axioms.

The sentences are controlled English, not free paraphrases. Named classes and
properties use ``rdfs:label`` when the ontology provides one, otherwise the
IRI local name. Complex descriptions fall back to a Manchester-style phrase.
Class names are copied verbatim: this module does not pluralise them.
"""

from __future__ import annotations

from collections.abc import Mapping

from ollab.core.ontology import Axiom, AxiomKind, Expr

_CARD_EN = {"min": "at least", "max": "at most", "exactly": "exactly"}
_CARD_MANCHESTER = {"min": "min", "max": "max", "exactly": "exactly"}


def verbalize(axiom: Axiom, labels: Mapping[str, str] | None = None) -> str:
    """Return one English sentence for ``axiom``.

    ``labels`` maps IRIs to display strings. Missing entries use the local name.
    """
    names = labels or {}
    if axiom.source == "subClassOf" and axiom.left is not None and axiom.right is not None:
        simple = _simple_subclass(axiom.left, axiom.right, names)
        if simple is not None:
            return simple
        return f"Every {manchester(axiom.left, names)} is ({manchester(axiom.right, names)})."
    if axiom.source == "equivalentClass" and axiom.left is not None and axiom.right is not None:
        if _is_named(axiom.left) and _is_named(axiom.right):
            return f"{_name(axiom.left, names)} is equivalent to {_name(axiom.right, names)}."
        return f"{manchester(axiom.left, names)} is equivalent to {manchester(axiom.right, names)}."
    if axiom.source == "disjointWith" and axiom.left is not None and axiom.right is not None:
        left = _name(axiom.left, names) if _is_named(axiom.left) else manchester(axiom.left, names)
        right = (
            _name(axiom.right, names) if _is_named(axiom.right) else manchester(axiom.right, names)
        )
        return f"{left} is disjoint with {right}."
    if axiom.source == "AllDisjointClasses" and axiom.members:
        return f"{_join_members(axiom.members, names)} are pairwise disjoint."
    if axiom.source == "domain" and axiom.prop and axiom.right is not None:
        return f"{_iri(axiom.prop, names)} has domain {_target(axiom.right, names)}."
    if axiom.source == "range" and axiom.prop and axiom.right is not None:
        return f"{_iri(axiom.prop, names)} has range {_target(axiom.right, names)}."
    if axiom.kind is AxiomKind.FUNCTIONAL and axiom.prop:
        if axiom.source == "inverseFunctional":
            return f"{_iri(axiom.prop, names)} is inverse-functional."
        return f"{_iri(axiom.prop, names)} is functional."
    return "unsupported-expression."


def manchester(expr: Expr, labels: Mapping[str, str] | None = None) -> str:
    """Render ``expr`` in a small Manchester fragment."""
    names = labels or {}
    if expr.kind == "iri" and expr.iri:
        return _iri(expr.iri, names)
    if expr.kind == "literal":
        return expr.value or "unsupported-expression"
    if expr.kind == "some" and expr.prop and expr.filler is not None:
        return f"{_iri(expr.prop, names)} some {manchester(expr.filler, names)}"
    if expr.kind == "only" and expr.prop and expr.filler is not None:
        return f"{_iri(expr.prop, names)} only {manchester(expr.filler, names)}"
    if expr.kind == "value" and expr.prop:
        target = manchester(expr.filler, names) if expr.filler is not None else expr.value
        if not target:
            return "unsupported-expression"
        return f"{_iri(expr.prop, names)} value {target}"
    if expr.kind == "card" and expr.prop and expr.cardinality and expr.n is not None:
        prop = _iri(expr.prop, names)
        word = _CARD_MANCHESTER[expr.cardinality]
        if expr.filler is None:
            return f"{prop} {word} {expr.n}"
        return f"{prop} {word} {expr.n} {manchester(expr.filler, names)}"
    if expr.kind == "and" and expr.operands:
        return "(" + " and ".join(manchester(item, names) for item in expr.operands) + ")"
    if expr.kind == "or" and expr.operands:
        return "(" + " or ".join(manchester(item, names) for item in expr.operands) + ")"
    if expr.kind == "not" and expr.filler is not None:
        return f"not ({manchester(expr.filler, names)})"
    if expr.kind == "one_of" and expr.operands:
        inner = ", ".join(manchester(item, names) for item in expr.operands)
        return "{" + inner + "}"
    return "unsupported-expression"


def local_name(iri: str) -> str:
    """Return the fragment after ``#``, or the last path segment."""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]


def _simple_subclass(left: Expr, right: Expr, labels: Mapping[str, str]) -> str | None:
    if not _is_named(left):
        return None
    subject = _name(left, labels)
    if _is_named(right):
        return f"Every {subject} is {_article(_name(right, labels))}."
    if right.kind in {"some", "only"} and right.prop and _is_named(right.filler):
        relation = "some" if right.kind == "some" else "only"
        filler = _name(right.filler, labels)
        return f"Every {subject} relates via {_iri(right.prop, labels)} to {relation} {filler}."
    if right.kind == "value" and right.prop:
        target = _name(right.filler, labels) if _is_named(right.filler) else right.value
        if target:
            prop = _iri(right.prop, labels)
            return f"Every {subject} relates via {prop} to the value {target}."
    if (
        right.kind == "card"
        and right.prop
        and right.cardinality
        and right.n is not None
        and (_is_named(right.filler) or right.filler is None)
    ):
        amount = f"{_CARD_EN[right.cardinality]} {right.n}"
        prop = _iri(right.prop, labels)
        if right.filler is None:
            return f"Every {subject} relates via {prop} to {amount} things."
        return f"Every {subject} relates via {prop} to {amount} {_name(right.filler, labels)}."
    return None


def _join_members(members: tuple[Expr, ...], labels: Mapping[str, str]) -> str:
    parts = [_target(item, labels) for item in members]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _target(expr: Expr, labels: Mapping[str, str]) -> str:
    if _is_named(expr):
        return _name(expr, labels)
    return manchester(expr, labels)


def _is_named(expr: Expr | None) -> bool:
    return expr is not None and expr.kind == "iri" and bool(expr.iri)


def _name(expr: Expr, labels: Mapping[str, str]) -> str:
    if expr.iri is None:
        return "unsupported-expression"
    return _iri(expr.iri, labels)


def _iri(iri: str, labels: Mapping[str, str]) -> str:
    label = labels.get(iri, "").strip()
    if label:
        return label
    return local_name(iri)


def _article(phrase: str) -> str:
    """Orthographic a/an from the first character. This is not a phonetic rule."""
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    return f"{article} {phrase}"
