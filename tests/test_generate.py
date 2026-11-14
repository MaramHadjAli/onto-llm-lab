"""A0/A1 prompts, Turtle parsing, and undefined terms. MockLLM only."""

from pathlib import Path

from rdflib import Graph

from ollab.core.llm import LLMRequest, hash_prompt, make_client
from ollab.core.ontology import load_ontology
from ollab.core.verbalize import verbalize
from ollab.kgen.generate import (
    axioms_for_class,
    build_prompt,
    extract_turtle,
    generate_individuals,
    prompt_labels,
    select_classes,
)
from tests.support import MINI, ONTOLOGIES

NS = "http://example.org/mini#"


def test_replicate_index_is_not_part_of_the_prompt(tmp_path: Path) -> None:
    prompts: list[str] = []

    def respond(request: LLMRequest) -> str:
        prompts.append(request.prompt)
        return "ind:1 a :Pizza .\n"

    client = make_client("mock", cache_dir=tmp_path / "cache", responder=respond)
    ontology = load_ontology(MINI)
    target = f"{NS}Pizza"
    shared = dict(
        target_class=target,
        condition="A0",
        n_individuals=1,
        seed=0,
        model="mock-1",
        temperature=0.0,
        max_tokens=64,
        prompt_version="v3",
    )
    first = generate_individuals(ontology, str(MINI), client, replicate_index=0, **shared)
    second = generate_individuals(ontology, str(MINI), client, replicate_index=1, **shared)
    assert prompts[0] == prompts[1]
    assert "replicate" not in prompts[0].lower()
    assert first.prompt_hash == second.prompt_hash
    assert first.cached is False and second.cached is False
    assert first.replicate_index == 0 and second.replicate_index == 1


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
    labels = prompt_labels(ontology)
    sentence = next(
        verbalize(axiom, labels)
        for axiom in axioms_for_class(ontology, target)
        if "CheesyPizza" in verbalize(axiom, labels)
    )
    a0 = build_prompt(ontology, target, condition="A0", n_individuals=3, seed=0)
    a1 = build_prompt(ontology, target, condition="A1", n_individuals=3, seed=0)
    assert "Axioms:" not in a0
    assert sentence not in a0
    assert sentence in a1
    assert a1.startswith(a0)
    assert a1[len(a0) :].startswith("\nAxioms:\n")
    assert "Cheesy pizza" not in a0
    assert ":CheesyPizza" in a0


def test_prompt_uses_english_labels_curies_and_declared_prefixes() -> None:
    ontology = load_ontology(MINI)
    target = f"{NS}NamedPizza"
    prompt = build_prompt(ontology, target, condition="A0", n_individuals=3, seed=0)
    assert prompt.startswith("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
    assert "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ." in prompt
    assert "@prefix owl: <http://www.w3.org/2002/07/owl#> ." in prompt
    assert "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ." in prompt
    assert f"@prefix : <{NS}> ." in prompt
    assert "@prefix ind: <http://ollab.local/gen/0/NamedPizza/> ." in prompt
    assert "Named pizza" in prompt
    assert "Pizza nomeada" not in prompt
    assert "Individual IRI pattern: ind:N" in prompt
    classes = prompt.split("Classes:\n", 1)[1].split("\nProperties:", 1)[0]
    properties = prompt.split("Properties:\n", 1)[1]
    assert "<" not in classes
    assert "<" not in properties
    assert ":NamedPizza" in classes
    assert ":hasTopping" in properties


def test_colliding_labels_fall_back_to_local_names(tmp_path: Path) -> None:
    path = tmp_path / "collision.ttl"
    path.write_text(
        "\n".join(
            [
                "@prefix : <http://example.org/col#> .",
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "<http://example.org/col> a owl:Ontology .",
                ':Alpha a owl:Class ; rdfs:label "Same"@en .',
                ':Beta a owl:Class ; rdfs:label "Same"@en .',
                ':Gamma a owl:Class ; rdfs:label "Outro"@pt .',
                ':Delta a owl:Class ; rdfs:label "British name"@en-GB .',
            ]
        ),
        encoding="utf-8",
    )
    ontology = load_ontology(path)
    prompt = build_prompt(
        ontology,
        "http://example.org/col#Alpha",
        condition="A0",
        n_individuals=1,
        seed=0,
    )
    assert "Same" not in prompt
    assert "Outro" not in prompt
    assert "British name" in prompt
    assert "- Alpha :Alpha" in prompt
    assert "- Beta :Beta" in prompt
    assert "- Gamma :Gamma" in prompt


def test_pizza_prompt_hides_non_english_labels() -> None:
    ontology = load_ontology(ONTOLOGIES / "pizza.owl")
    base = "https://raw.githubusercontent.com/owlcs/pizza-ontology/refs/heads/master/pizza.owl#"
    target = base + "PizzaBase"
    a0 = build_prompt(ontology, target, condition="A0", n_individuals=3, seed=0)
    a1 = build_prompt(ontology, target, condition="A1", n_individuals=3, seed=0)
    assert "BaseDaPizza" not in a0
    assert "CoberturaDeQueijo" not in a0
    assert "- PizzaBase :PizzaBase" in a0
    assert f"@prefix : <{base}> ." in a0
    assert a1.startswith(a0)
    assert "\nAxioms:\n" in a1[len(a0) :]


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


_V1_TYPE = "Every individual must have an explicit rdf:type."
_V1_OUTPUT = "Output Turtle only, with no markdown fence and no explanation."
_V2_TYPE = (
    "Describe each individual with an explicit rdf:type and with property assertions "
    "using the listed properties where they apply. Every individual used as the object "
    "of a property must also be declared with an explicit rdf:type."
)
_V2_OUTPUT = (
    "Output only the Turtle triples. Do not write @prefix lines and do not use a "
    "markdown fence: the prefixes above are already declared."
)
# prompt_hash of the v1 A0 PizzaBase prompt, seed 0, 3 individuals, from the 7B smoke.
_V1_PIZZA_BASE_A0 = "31fee5ecdd29f39262cf444a39caecc05c93e651560848486659a8820bfb247c"
_V1_PIZZA_BASE_A1 = "b1a87ec5e9a6d12dc4c5ac5505d7ec13aa68478149de50450bf798f216f55d53"
_PIZZA = "https://raw.githubusercontent.com/owlcs/pizza-ontology/refs/heads/master/pizza.owl#"


def test_v1_prompt_text_and_hash_match_the_recorded_smoke() -> None:
    ontology = load_ontology(ONTOLOGIES / "pizza.owl")
    target = _PIZZA + "PizzaBase"
    a0 = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=0, prompt_version="v1"
    )
    a1 = build_prompt(
        ontology, target, condition="A1", n_individuals=3, seed=0, prompt_version="v1"
    )
    assert _V1_TYPE in a0
    assert _V1_OUTPUT in a0
    assert "\nTarget class: PizzaBase :PizzaBase\n" in a0
    assert _V2_OUTPUT not in a0
    assert a1.startswith(a0)
    assert a1[len(a0) :].startswith("\nAxioms:\n")
    assert hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=a0, prompt_version="v1")) == (
        _V1_PIZZA_BASE_A0
    )
    assert hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=a1, prompt_version="v1")) == (
        _V1_PIZZA_BASE_A1
    )
    bare = hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=a0))
    assert bare == _V1_PIZZA_BASE_A0


def test_v2_changes_only_the_three_instruction_lines() -> None:
    ontology = load_ontology(ONTOLOGIES / "pizza.owl")
    target = _PIZZA + "PizzaBase"
    a0 = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=0, prompt_version="v2"
    )
    a1 = build_prompt(
        ontology, target, condition="A1", n_individuals=3, seed=0, prompt_version="v2"
    )
    assert _V2_TYPE in a0
    assert _V2_OUTPUT in a0
    assert _V1_TYPE not in a0
    assert _V1_OUTPUT not in a0
    assert "\nTarget class: :PizzaBase\n" in a0
    assert "\nTarget class: PizzaBase :PizzaBase\n" not in a0
    assert a1.startswith(a0)
    assert a1[len(a0) :].startswith("\nAxioms:\n")
    v2_hash = hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=a0, prompt_version="v2"))
    assert v2_hash != hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=a0))
    assert v2_hash != _V1_PIZZA_BASE_A0


def test_prefix_completion_parse_metrics() -> None:
    ontology = load_ontology(MINI)
    target = f"{NS}Pizza"

    def run(text: str):
        return generate_individuals(
            ontology,
            str(MINI),
            make_client("mock", responder=lambda _request: text),
            target_class=target,
            condition="A0",
            n_individuals=2,
            seed=0,
            model="mock-1",
        )

    ind_style = "ind:1 a :Pizza ;\n    :hasTopping ind:2 .\nind:2 a :PizzaTopping .\n"
    bare = run(ind_style)
    assert bare.prompt_version == "v2"
    assert bare.strict_parse_ok is False
    assert bare.lenient_parse_ok is False
    assert bare.prefix_completed_parse_ok is True
    assert bare.parse_error is None
    assert bare.n_individuals == 2
    assert bare.n_type_triples == 2
    assert bare.n_property_triples == 1
    assert f"{NS}Pizza" not in bare.undefined_terms
    parsed = Graph()
    parsed.parse(data=bare.turtle, format="turtle")
    assert "http://ollab.local/gen/0/Pizza/1" in {str(subject) for subject in parsed.subjects()}

    declared = (
        f"@prefix : <{NS}> .\n@prefix ind: <http://ollab.local/gen/0/Pizza/> .\nind:1 a :Pizza .\n"
    )
    already = run(declared)
    assert already.strict_parse_ok is True
    assert already.lenient_parse_ok is True
    assert already.prefix_completed_parse_ok is True
    assert already.n_type_triples == 1
    assert already.n_property_triples == 0

    fenced = run("```turtle\nind:1 a :Pizza .\n```\n")
    assert fenced.strict_parse_ok is False
    assert fenced.lenient_parse_ok is False
    assert fenced.prefix_completed_parse_ok is True
    assert fenced.n_type_triples == 1

    prose = run("this is not turtle")
    assert prose.strict_parse_ok is False
    assert prose.lenient_parse_ok is False
    assert prose.prefix_completed_parse_ok is False
    assert prose.parse_error
    assert prose.turtle is None
    assert prose.n_individuals == 0
    assert prose.n_type_triples == 0
    assert prose.n_property_triples == 0


def test_v3_removes_only_the_seed_line() -> None:
    ontology = load_ontology(ONTOLOGIES / "pizza.owl")
    target = _PIZZA + "PizzaBase"
    v1 = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=0, prompt_version="v1"
    )
    v2 = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=0, prompt_version="v2"
    )
    v3 = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=0, prompt_version="v3"
    )
    other = build_prompt(
        ontology, target, condition="A0", n_individuals=3, seed=1, prompt_version="v3"
    )
    assert v2.count("\nSeed: 0\n") == 1
    assert "\nSeed: 0\n" in v1
    assert v2.replace("\nSeed: 0\n", "\n", 1) == v3
    assert "Seed:" not in v3
    assert "http://ollab.local/gen/0/PizzaBase/" in v3
    assert "http://ollab.local/gen/1/PizzaBase/" in other
    assert v3 != other
    assert hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=v3, prompt_version="v3")) != (
        hash_prompt(LLMRequest(model="qwen2.5:7b", prompt=v2, prompt_version="v2"))
    )
