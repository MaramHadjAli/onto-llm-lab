"""A0 and A1 individual generation.

A0 puts class and property names in the prompt. A1 adds the verbalised
axioms that mention the target class, plus domain, range, and functional
axioms of the properties used in those axioms. The two prompts are identical
except for that axioms block. A name shown to the model is an English
``rdfs:label`` when the ontology has one, otherwise the IRI local name.
A display string shared by two IRIs is replaced by the local name. The prompt
declares the ``rdf``, ``rdfs``, ``owl``, ``xsd``, ontology, and individual
prefixes, and it writes classes and properties as CURIEs.

Prompts are deterministic for a fixed ontology, class, seed, and individual
count. Target classes are chosen once from a seeded sample of the named
classes, then sorted by IRI. Fresh individual IRIs are allowed. A predicate,
a ``rdf:type`` object, or a non-individual object IRI that is outside the
ontology and outside the RDF, RDFS, OWL, and XSD namespaces is an undefined
term.

Output parsing is the same for A0 and A1 and for both prompt versions.
``strict_parse_ok`` is true when the raw completion parses as Turtle.
``lenient_parse_ok`` is true when the text parses after :func:`extract_turtle`.
That function is the only repair of the completion: it drops a markdown
fence, including an unclosed one, and keeps the first Turtle block. It does
not add prefixes, insert punctuation, or rewrite IRIs. A separate step then
prepends the prompt's exact ``@prefix`` block. ``prefix_completed_parse_ok``
is true when that text parses. Validation uses the prefix-completed graph.

Prompt ``v1`` is the original wording. Prompt ``v2`` changes the output
instruction, the typing instruction, and the target-class line. A0 and A1
still differ only by the axioms block. See ``docs/prompt_versions.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from rdflib import Graph, URIRef
from rdflib import Literal as RdfLiteral
from rdflib.namespace import RDF, RDFS

from ollab.core.llm import LLMClient, LLMRequest
from ollab.core.ontology import Axiom, Expr, Ontology
from ollab.core.verbalize import local_name, verbalize

Condition = Literal["A0", "A1"]
PromptVersion = Literal["v1", "v2"]
_DECLARED_PREFIXES = (
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
)
_BUILTIN_PREFIXES = tuple(item[1] for item in _DECLARED_PREFIXES) + (
    "http://www.w3.org/XML/1998/namespace",
)
_EXCLUDED = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://www.w3.org/2002/07/owl#Nothing",
}


class GenerationRecord(BaseModel):
    """One class, one seed, one condition.

    ``strict_parse_ok`` is the raw completion. ``lenient_parse_ok`` is the
    result after ``extract_turtle``. ``prefix_completed_parse_ok`` is that
    text with the prompt's prefix block prepended. ``parse_error`` is the
    prefix-completed parse error, and is null when that parse succeeds.
    ``n_type_triples`` counts ``rdf:type`` triples in the prefix-completed
    graph. ``n_property_triples`` counts every other triple. Both are zero
    when the prefix-completed parse fails.
    """

    model_config = ConfigDict(frozen=True)

    condition: Condition
    seed: int
    target_class: str
    provider: str
    model: str
    temperature: float
    prompt_version: PromptVersion = "v2"
    prompt_hash: str
    strict_parse_ok: bool
    lenient_parse_ok: bool
    prefix_completed_parse_ok: bool
    parse_error: str | None
    n_requested: int
    n_individuals: int
    n_type_triples: int
    n_property_triples: int
    undefined_terms: tuple[str, ...]
    turtle: str | None
    raw_text: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None


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
    prompt_version: PromptVersion = "v2",
) -> str:
    """Build the A0 or A1 user prompt. The system message is unused.

    A1 is the A0 text plus the axioms block. Nothing earlier in the prompt
    depends on the condition. ``v1`` and ``v2`` share that split; the version
    only changes the three instruction lines described in
    ``docs/prompt_versions.md``.
    """
    _require_prompt_version(prompt_version)
    labels = prompt_labels(ontology)
    ontology_ns, individual_ns = _prompt_namespaces(ontology, target_class, seed)
    prefixes = _prefix_map(ontology_ns, individual_ns)
    lines = _prefix_lines(ontology_ns, individual_ns)
    lines.extend(
        _instruction_lines(
            prompt_version,
            n_individuals=n_individuals,
            seed=seed,
            target_class=target_class,
            labels=labels,
            prefixes=prefixes,
        )
    )
    for iri in ontology.named_classes():
        lines.append(f"- {_label(iri, labels)} {_curie(iri, prefixes)}")
    lines.append("")
    lines.append("Properties:")
    properties = ontology.object_properties() + ontology.data_properties()
    for iri in properties:
        lines.append(f"- {_label(iri, labels)} {_curie(iri, prefixes)}")
    shared = "\n".join(lines) + "\n"
    if condition == "A0":
        return shared
    axiom_lines = ["", "Axioms:"]
    axioms = axioms_for_class(ontology, target_class)
    if not axioms:
        axiom_lines.append("- No verbalised axiom mentions this class.")
    for index, axiom in enumerate(axioms, start=1):
        axiom_lines.append(f"{index}. {verbalize(axiom, labels)}")
    return shared + "\n".join(axiom_lines) + "\n"


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
    prompt_version: PromptVersion = "v2",
) -> GenerationRecord:
    """Call the shared LLM client once and score the Turtle it returns.

    ``ontology_path`` is recorded by the caller in the run file. It is accepted
    here so a record can be built without a second lookup. The prefix block
    prepended before the third parse is the same block the prompt starts with.
    """
    del ontology_path
    _require_prompt_version(prompt_version)
    prompt = build_prompt(
        ontology,
        target_class,
        condition=condition,
        n_individuals=n_individuals,
        seed=seed,
        prompt_version=prompt_version,
    )
    response = client.complete(
        LLMRequest(
            model=model,
            prompt=prompt,
            prompt_version=prompt_version,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
    )
    raw = response.text
    _, strict_error, _ = _parse_graph(raw)
    extracted = extract_turtle(raw)
    _, lenient_error, _ = _parse_graph(extracted)
    ontology_ns, individual_ns = _prompt_namespaces(ontology, target_class, seed)
    completed = _prefix_block(ontology_ns, individual_ns) + extracted
    turtle, completed_error, parsed = _parse_graph(completed)
    undefined: tuple[str, ...] = ()
    count = 0
    n_type = 0
    n_property = 0
    if parsed is not None:
        undefined = undefined_terms(ontology, parsed)
        count = _individual_count(parsed)
        n_type, n_property = _triple_counts(parsed)
    return GenerationRecord(
        condition=condition,
        seed=seed,
        target_class=target_class,
        provider=response.provider,
        model=response.model,
        temperature=temperature,
        prompt_version=prompt_version,
        prompt_hash=response.prompt_hash,
        strict_parse_ok=strict_error is None,
        lenient_parse_ok=lenient_error is None,
        prefix_completed_parse_ok=completed_error is None,
        parse_error=completed_error,
        n_requested=n_individuals,
        n_individuals=count,
        n_type_triples=n_type,
        n_property_triples=n_property,
        undefined_terms=undefined,
        turtle=turtle if completed_error is None else None,
        raw_text=raw,
        prompt_eval_count=response.prompt_eval_count,
        eval_count=response.eval_count,
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


def extract_turtle(text: str) -> str:
    """Return the first Turtle block, with its markdown fence removed.

    An opening fence is ````` `` plus an optional language tag on that line.
    A later ````` `` closes it. If the fence is unclosed, the block is the
    remainder of the text. Text before the first fence is discarded. When a
    block is tagged ``turtle`` or ``ttl``, the first such block is used;
    otherwise the first fence is used. With no fence, the stripped text is
    returned unchanged.

    This is the only repair applied to model output. It does not add
    prefixes, insert punctuation, or rewrite IRIs. A0 and A1 both call it.
    """
    stripped = text.strip()
    spans = _fence_spans(stripped)
    if not spans:
        return stripped
    for tag, body in spans:
        if tag.lower() in {"turtle", "ttl"}:
            return body.strip()
    return spans[0][1].strip()


def _parse_graph(text: str) -> tuple[str | None, str | None, Graph | None]:
    """Parse ``text`` as Turtle. The text is not modified before parsing."""
    graph = Graph()
    try:
        graph.parse(data=text, format="turtle")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    serialized = graph.serialize(format="turtle")
    if not isinstance(serialized, str):
        serialized = serialized.decode("utf-8")
    return serialized, None, graph


def _fence_spans(text: str) -> list[tuple[str, str]]:
    """Return ``(language tag, body)`` for each markdown fence, in order.

    The last fence may be unclosed. Its body then runs to the end of ``text``.
    """
    spans: list[tuple[str, str]] = []
    cursor = 0
    while True:
        start = text.find("```", cursor)
        if start < 0:
            return spans
        line_end = text.find("\n", start)
        if line_end < 0:
            spans.append((text[start + 3 :].strip(), ""))
            return spans
        tag = text[start + 3 : line_end].strip()
        close = text.find("```", line_end + 1)
        if close < 0:
            spans.append((tag, text[line_end + 1 :]))
            return spans
        spans.append((tag, text[line_end + 1 : close]))
        cursor = close + 3


def _triple_counts(graph: Graph) -> tuple[int, int]:
    """Return ``(rdf:type triples, every other triple)``."""
    n_type = 0
    n_property = 0
    for _subject, predicate, _obj in graph:
        if predicate == RDF.type:
            n_type += 1
        else:
            n_property += 1
    return n_type, n_property


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


def prompt_labels(ontology: Ontology) -> dict[str, str]:
    """Names for one prompt: an English label, otherwise the local name.

    Labels with no language tag and labels in any other language are ignored.
    When two IRIs would show the same string, both use the local name. When
    those local names still collide, the IRI itself is shown.
    """
    english = _english_labels(ontology.graph)
    iris = set(ontology.named_classes())
    iris.update(ontology.object_properties())
    iris.update(ontology.data_properties())
    for axiom in ontology.axioms():
        iris.update(_axiom_iris(axiom))
    preferred = {iri: english.get(iri) or local_name(iri) for iri in iris}
    return _unique_labels(preferred)


def _english_labels(graph: Graph) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for subject, obj in graph.subject_objects(RDFS.label):
        if not isinstance(subject, URIRef) or not isinstance(obj, RdfLiteral):
            continue
        if not _is_english(obj.language or ""):
            continue
        text = str(obj).strip()
        if not text:
            continue
        grouped.setdefault(str(subject), []).append(text)
    return {iri: sorted(texts)[0] for iri, texts in grouped.items()}


def _is_english(language: str) -> bool:
    tag = language.lower()
    return tag == "en" or tag.startswith("en-")


def _unique_labels(preferred: dict[str, str]) -> dict[str, str]:
    counts = Counter(preferred.values())
    resolved = {
        iri: local_name(iri) if counts[text] > 1 else text for iri, text in preferred.items()
    }
    counts = Counter(resolved.values())
    if all(count == 1 for count in counts.values()):
        return resolved
    again = {iri: local_name(iri) if counts[text] > 1 else text for iri, text in resolved.items()}
    counts = Counter(again.values())
    if all(count == 1 for count in counts.values()):
        return again
    return {iri: iri if counts[text] > 1 else text for iri, text in again.items()}


def _ontology_namespace(ontology: Ontology) -> str:
    iris = ontology.named_classes() or ontology.object_properties() or ontology.data_properties()
    if not iris:
        if ontology.iri:
            return _namespace_of(ontology.iri)
        raise ValueError("ontology has no named class, property, or IRI")
    counts = Counter(_namespace_of(iri) for iri in iris)
    return sorted(counts, key=lambda ns: (-counts[ns], ns))[0]


def _namespace_of(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    if "/" in iri:
        return iri.rsplit("/", 1)[0] + "/"
    return iri


def _require_prompt_version(prompt_version: str) -> None:
    if prompt_version not in ("v1", "v2"):
        raise ValueError(f"prompt_version must be 'v1' or 'v2', got {prompt_version!r}")


def _instruction_lines(
    prompt_version: PromptVersion,
    *,
    n_individuals: int,
    seed: int,
    target_class: str,
    labels: dict[str, str],
    prefixes: dict[str, str],
) -> list[str]:
    """The shared instruction block. v1 text is the original wording."""
    if prompt_version == "v1":
        type_line = "Every individual must have an explicit rdf:type."
        output_line = "Output Turtle only, with no markdown fence and no explanation."
        target_line = (
            f"Target class: {_label(target_class, labels)} {_curie(target_class, prefixes)}"
        )
    else:
        type_line = (
            "Describe each individual with an explicit rdf:type and with property "
            "assertions using the listed properties where they apply. Every individual "
            "used as the object of a property must also be declared with an explicit "
            "rdf:type."
        )
        output_line = (
            "Output only the Turtle triples. Do not write @prefix lines and do not use "
            "a markdown fence: the prefixes above are already declared."
        )
        target_line = f"Target class: {_curie(target_class, prefixes)}"
    return [
        "",
        f"Generate {n_individuals} RDF individuals as Turtle.",
        "Use only the class and property IRIs listed below, plus new individual IRIs.",
        type_line,
        output_line,
        f"Seed: {seed}",
        target_line,
        "Individual IRI pattern: ind:N",
        "",
        "Classes:",
    ]


def _prompt_namespaces(ontology: Ontology, target_class: str, seed: int) -> tuple[str, str]:
    ontology_ns = _ontology_namespace(ontology)
    individual_ns = f"http://ollab.local/gen/{seed}/{local_name(target_class)}/"
    return ontology_ns, individual_ns


def _prefix_block(ontology_ns: str, individual_ns: str) -> str:
    """The prompt's ``@prefix`` lines, including the trailing newline.

    This is the block prepended to a completion. It is not a repair of the
    model text: the completion is left unchanged and the same bytes are used
    for every condition and every prompt version.
    """
    return "\n".join(_prefix_lines(ontology_ns, individual_ns)) + "\n"


def _prefix_lines(ontology_ns: str, individual_ns: str) -> list[str]:
    lines = [f"@prefix {name}: <{ns}> ." for name, ns in _DECLARED_PREFIXES]
    lines.append(f"@prefix : <{ontology_ns}> .")
    lines.append(f"@prefix ind: <{individual_ns}> .")
    return lines


def _prefix_map(ontology_ns: str, individual_ns: str) -> dict[str, str]:
    mapping = {ns: name for name, ns in _DECLARED_PREFIXES}
    mapping[ontology_ns] = ""
    mapping[individual_ns] = "ind"
    return mapping


def _curie(iri: str, prefixes: dict[str, str]) -> str:
    matches = [ns for ns in prefixes if iri.startswith(ns)]
    if not matches:
        return f"<{iri}>"
    ns = max(matches, key=len)
    name = prefixes[ns]
    local = iri[len(ns) :]
    if name:
        return f"{name}:{local}"
    return f":{local}"


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
