"""A0 and A1 individual generation.

A0 puts class and property labels in the prompt. A1 adds the verbalised
axioms that mention the target class, plus domain, range, and functional
axioms of the properties used in those axioms. Prompts are deterministic for
a fixed ontology, class, seed, and individual count.

Target classes are chosen once from a seeded sample of the named classes,
then sorted by IRI. Fresh individual IRIs are allowed. A predicate, a
``rdf:type`` object, or a non-individual object IRI that is outside the
ontology and outside the RDF, RDFS, OWL, and XSD namespaces is an undefined term.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from ollab.core.llm import LLMClient, LLMRequest
from ollab.core.ontology import Axiom, Expr, Ontology
from ollab.core.verbalize import local_name, verbalize

Condition = Literal["A0", "A1"]
_FENCE = re.compile(r"```(?:turtle|ttl)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BUILTIN_PREFIXES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/XML/1998/namespace",
)
_EXCLUDED = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://www.w3.org/2002/07/owl#Nothing",
}


class GenerationRecord(BaseModel):
    """One class, one seed, one condition."""

    model_config = ConfigDict(frozen=True)

    condition: Condition
    seed: int
    target_class: str
    provider: str
    model: str
    temperature: float
    prompt_hash: str
    parse_ok: bool
    parse_error: str | None
    n_requested: int
    n_individuals: int
    undefined_terms: tuple[str, ...]
    turtle: str | None
    raw_text: str


def select_classes(ontology: Ontology, n: int, seed: int) -> tuple[str, ...]:
    """Return ``n`` named class IRIs.

    The sample is drawn with ``numpy.random.default_rng(seed)`` and then
    sorted, so the set depends on the seed and the order does not.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    pool = tuple(
        iri
        for iri in ontology.named_classes()
        if iri not in _EXCLUDED and not iri.endswith("#Thing") and not iri.endswith("#Nothing")
    )
    if n > len(pool):
        raise ValueError(f"requested {n} classes but the ontology has {len(pool)}")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(pool), size=n, replace=False)
    return tuple(sorted(pool[int(index)] for index in chosen))


def build_prompt(
    ontology: Ontology,
    target_class: str,
    *,
    condition: Condition,
    n_individuals: int,
    seed: int,
) -> str:
    """Build the A0 or A1 user prompt. The system message is unused."""
    labels = ontology.labels()
    lines = [
        f"Generate {n_individuals} RDF individuals as Turtle.",
        "Use only the class and property IRIs listed below, plus new individual IRIs.",
        "Every individual must have an explicit rdf:type.",
        "Output Turtle only, with no markdown fence and no explanation.",
        f"Seed: {seed}",
        f"Target class: {_label(target_class, labels)} <{target_class}>",
        f"Individual IRI pattern: <http://ollab.local/gen/{seed}/{local_name(target_class)}/N>",
        "",
        "Classes:",
    ]
    for iri in ontology.named_classes():
        lines.append(f"- {_label(iri, labels)} <{iri}>")
    lines.append("")
    lines.append("Properties:")
    properties = ontology.object_properties() + ontology.data_properties()
    for iri in properties:
        lines.append(f"- {_label(iri, labels)} <{iri}>")
    if condition == "A1":
        lines.append("")
        lines.append("Axioms:")
        axioms = axioms_for_class(ontology, target_class)
        if not axioms:
            lines.append("- No verbalised axiom mentions this class.")
        for index, axiom in enumerate(axioms, start=1):
            lines.append(f"{index}. {verbalize(axiom, labels)}")
    return "\n".join(lines) + "\n"


def axioms_for_class(ontology: Ontology, target_class: str) -> tuple[Axiom, ...]:
    """Axioms that mention ``target_class``, plus constraints on their properties."""
    properties = set(ontology.object_properties()) | set(ontology.data_properties())
    direct: list[Axiom] = []
    used_properties: set[str] = set()
    for axiom in ontology.axioms():
        iris = _axiom_iris(axiom)
        props = _axiom_properties(axiom) & properties
        if target_class in iris:
            direct.append(axiom)
            used_properties.update(props)
    extra = [
        axiom
        for axiom in ontology.axioms()
        if axiom.axiom_id not in {item.axiom_id for item in direct}
        and axiom.prop in used_properties
        and axiom.kind.value in {"Domain", "Range", "Functional"}
    ]
    chosen = {axiom.axiom_id: axiom for axiom in direct}
    for axiom in extra:
        chosen.setdefault(axiom.axiom_id, axiom)
    return tuple(sorted(chosen.values(), key=lambda item: item.axiom_id))


def generate_individuals(
    ontology: Ontology,
    ontology_path: str,
    client: LLMClient,
    *,
    target_class: str,
    condition: Condition,
    n_individuals: int,
    seed: int,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> GenerationRecord:
    """Call the shared LLM client once and score the Turtle it returns.

    ``ontology_path`` is recorded by the caller in the run file. It is accepted
    here so a record can be built without a second lookup.
    """
    del ontology_path
    prompt = build_prompt(
        ontology,
        target_class,
        condition=condition,
        n_individuals=n_individuals,
        seed=seed,
    )
    response = client.complete(
        LLMRequest(
            model=model,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
    )
    turtle, error, parsed = _parse_turtle(response.text)
    undefined: tuple[str, ...] = ()
    count = 0
    if parsed is not None:
        undefined = undefined_terms(ontology, parsed)
        count = _individual_count(parsed)
    return GenerationRecord(
        condition=condition,
        seed=seed,
        target_class=target_class,
        provider=response.provider,
        model=response.model,
        temperature=temperature,
        prompt_hash=response.prompt_hash,
        parse_ok=error is None,
        parse_error=error,
        n_requested=n_individuals,
        n_individuals=count,
        undefined_terms=undefined,
        turtle=turtle if error is None else None,
        raw_text=response.text,
    )


def undefined_terms(ontology: Ontology, data: Graph) -> tuple[str, ...]:
    """Return sorted IRIs used as vocabulary but absent from the ontology."""
    known = _ontology_iris(ontology)
    fresh = {str(subject) for subject in data.subjects() if isinstance(subject, URIRef)}
    bad: set[str] = set()
    for subject, predicate, obj in data:
        if isinstance(predicate, URIRef) and not _allowed(str(predicate), known):
            bad.add(str(predicate))
        if predicate == RDF.type and isinstance(obj, URIRef) and not _allowed(str(obj), known):
            bad.add(str(obj))
        if isinstance(obj, URIRef) and predicate != RDF.type:
            iri = str(obj)
            if iri not in fresh and not _allowed(iri, known):
                bad.add(iri)
        del subject
    return tuple(sorted(bad))


def _parse_turtle(text: str) -> tuple[str | None, str | None, Graph | None]:
    candidate = _extract_turtle(text)
    graph = Graph()
    try:
        graph.parse(data=candidate, format="turtle")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    serialized = graph.serialize(format="turtle")
    if not isinstance(serialized, str):
        serialized = serialized.decode("utf-8")
    return serialized, None, graph


def _extract_turtle(text: str) -> str:
    stripped = text.strip()
    fenced = _FENCE.search(stripped)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _individual_count(graph: Graph) -> int:
    subjects = {
        subject
        for subject, predicate, obj in graph.triples((None, RDF.type, None))
        if isinstance(subject, URIRef) and isinstance(obj, URIRef) and predicate == RDF.type
    }
    return len(subjects)


def _ontology_iris(ontology: Ontology) -> set[str]:
    found = {str(term) for term in ontology.graph.subjects() if isinstance(term, URIRef)}
    found.update(str(term) for term in ontology.graph.predicates() if isinstance(term, URIRef))
    found.update(str(term) for term in ontology.graph.objects() if isinstance(term, URIRef))
    return found


def _allowed(iri: str, known: set[str]) -> bool:
    if iri in known:
        return True
    return any(iri.startswith(prefix) for prefix in _BUILTIN_PREFIXES)


def _label(iri: str, labels: dict[str, str]) -> str:
    text = labels.get(iri, "").strip()
    return text or local_name(iri)


def _axiom_iris(axiom: Axiom) -> set[str]:
    found = _expr_iris(axiom.left) | _expr_iris(axiom.right)
    if axiom.prop:
        found.add(axiom.prop)
    for member in axiom.members:
        found |= _expr_iris(member)
    return found


def _axiom_properties(axiom: Axiom) -> set[str]:
    found = _expr_props(axiom.left) | _expr_props(axiom.right)
    if axiom.prop:
        found.add(axiom.prop)
    for member in axiom.members:
        found |= _expr_props(member)
    return found


def _expr_iris(expr: Expr | None) -> set[str]:
    if expr is None:
        return set()
    found: set[str] = set()
    if expr.iri:
        found.add(expr.iri)
    if expr.prop:
        found.add(expr.prop)
    found |= _expr_iris(expr.filler)
    for operand in expr.operands:
        found |= _expr_iris(operand)
    return found


def _expr_props(expr: Expr | None) -> set[str]:
    if expr is None:
        return set()
    found: set[str] = set()
    if expr.prop:
        found.add(expr.prop)
    found |= _expr_props(expr.filler)
    for operand in expr.operands:
        found |= _expr_props(operand)
    return found
