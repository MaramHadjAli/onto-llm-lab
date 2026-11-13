"""A0/A1 prompts, Turtle parsing, and undefined terms. MockLLM only."""

from ollab.core.llm import LLMRequest, make_client
from ollab.core.ontology import load_ontology
from ollab.core.verbalize import verbalize
from ollab.kgen.generate import (
    axioms_for_class,
    build_prompt,
    extract_turtle,
    generate_individuals,
    select_classes,
)
from tests.support import MINI

NS = "http://example.org/mini#"


def test_class_selection_is_deterministic() -> None:
    ontology = load_ontology(MINI)
    first = select_classes(ontology, 2, 0)
    assert first == select_classes(ontology, 2, 0)
    assert len(first) == 2
    assert first == tuple(sorted(first))
    named = set(ontology.named_classes())
    assert set(first) <= named
    variants = {select_classes(ontology, 2, seed) for seed in range(30)}
    assert len(variants) > 1


def test_a1_adds_verbalised_axioms_and_a0_does_not() -> None:
    ontology = load_ontology(MINI)
    target = f"{NS}CheesyPizza"
    labels = ontology.labels()
    sentence = next(
        verbalize(axiom, labels)
        for axiom in axioms_for_class(ontology, target)
        if "Cheesy pizza" in verbalize(axiom, labels)
    )
    a0 = build_prompt(ontology, target, condition="A0", n_individuals=3, seed=0)
    a1 = build_prompt(ontology, target, condition="A1", n_individuals=3, seed=0)
    assert "Axioms:" not in a0
    assert sentence not in a0
    assert sentence in a1
    assert "Cheesy pizza" in a0


def test_parse_failure_is_counted() -> None:
    ontology = load_ontology(MINI)
    client = make_client("mock", responder=lambda _request: "this is not turtle")
    record = generate_individuals(
        ontology,
        str(MINI),
        client,
        target_class=f"{NS}Pizza",
        condition="A0",
        n_individuals=2,
        seed=0,
        model="mock-1",
    )
    assert record.strict_parse_ok is False
    assert record.lenient_parse_ok is False
    assert record.parse_error
    assert record.n_individuals == 0
    assert record.provider == "mock"
    assert record.seed == 0
    assert record.prompt_hash


def test_undefined_property_is_reported_and_fenced_turtle_parses() -> None:
    ontology = load_ontology(MINI)

    def respond(_request: LLMRequest) -> str:
        return (
            "```turtle\n"
            f"@prefix : <{NS}> .\n"
            ":one a :Pizza .\n"
            ":two a :Pizza ; :notAProperty :one .\n"
            ":three a :InventedClass .\n"
            "```\n"
        )

    record = generate_individuals(
        ontology,
        str(MINI),
        make_client("mock", responder=respond),
        target_class=f"{NS}Pizza",
        condition="A0",
        n_individuals=3,
        seed=1,
        model="mock-1",
    )
    assert record.strict_parse_ok is False
    assert record.lenient_parse_ok is True
    assert record.parse_error is None
    assert record.n_individuals == 3
    assert f"{NS}notAProperty" in record.undefined_terms
    assert f"{NS}InventedClass" in record.undefined_terms
    assert f"{NS}Pizza" not in record.undefined_terms
    assert f"{NS}one" not in record.undefined_terms


def test_raw_turtle_is_strict_and_a_missing_dot_is_not_repaired() -> None:
    ontology = load_ontology(MINI)
    raw = f"@prefix : <{NS}> .\n:one a :Pizza .\n"

    def respond(_request: LLMRequest) -> str:
        return raw

    record = generate_individuals(
        ontology,
        str(MINI),
        make_client("mock", responder=respond),
        target_class=f"{NS}Pizza",
        condition="A0",
        n_individuals=1,
        seed=0,
        model="mock-1",
    )
    assert record.strict_parse_ok is True
    assert record.lenient_parse_ok is True
    broken = "```turtle\n:one a :Pizza\n```"
    assert extract_turtle(broken) == ":one a :Pizza"
    client = make_client("mock", responder=lambda _request: broken)
    failed = generate_individuals(
        ontology,
        str(MINI),
        client,
        target_class=f"{NS}Pizza",
        condition="A1",
        n_individuals=1,
        seed=0,
        model="mock-1",
    )
    assert failed.strict_parse_ok is False
    assert failed.lenient_parse_ok is False


def test_unclosed_fence_and_first_turtle_block() -> None:
    body = f"@prefix : <{NS}> .\n:one a :Pizza .\n"
    unclosed = "```turtle\n" + body
    assert extract_turtle(unclosed) == body.strip()
    wrapped = "note\n```json\n{ }\n```\n```ttl\n" + body + "```\n"
    assert extract_turtle(wrapped) == body.strip()
    ontology = load_ontology(MINI)
    flags = []
    for condition in ("A0", "A1"):
        record = generate_individuals(
            ontology,
            str(MINI),
            make_client("mock", responder=lambda _request: unclosed),
            target_class=f"{NS}Pizza",
            condition=condition,
            n_individuals=1,
            seed=2,
            model="mock-1",
        )
        flags.append((record.strict_parse_ok, record.lenient_parse_ok, record.n_individuals))
    assert flags == [(False, True, 1), (False, True, 1)]
