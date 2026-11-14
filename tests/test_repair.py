"""A2 reuses the A1 completion and stops when the feedback categories are clear."""

from ollab.core.llm import LLMRequest, make_client
from ollab.core.ontology import load_ontology
from ollab.kgen.generate import build_prompt, generate_individuals
from ollab.kgen.repair import RepairError, render_feedback, repair_prompt, run_a2, shacl_feedback
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_owl import check_consistency
from tests.support import MINI

NS = "http://example.org/mini#"
TARGET = f"{NS}Pizza"
VALID = "ind:1 a :Pizza .\n"


def test_shacl_feedback_deduplicates_sorts_and_caps() -> None:
    assert shacl_feedback(["b", "a", "b"]) == "- a\n- b"
    messages = [f"m{index}" for index in range(12)]
    text = shacl_feedback(messages)
    assert text.endswith("\n2 further distinct messages omitted.")
    assert "- m0" in text
    assert "- m8" not in text
    assert "- m9" not in text
    assert shacl_feedback([]) == "none"


def test_repair_prompt_keeps_the_a1_text_and_names_the_failures() -> None:
    ontology = load_ontology(MINI)
    a1 = build_prompt(
        ontology, TARGET, condition="A1", n_individuals=1, seed=0, prompt_version="v3"
    )
    feedback = render_feedback(
        parse_error="BadSyntax: broken",
        validation_ran=False,
        consistency=None,
        violations=None,
        undefined=(),
        raw_text="not turtle",
    )
    prompt = repair_prompt(a1, feedback)
    assert prompt.startswith(a1 + "\n")
    assert "Seed:" not in a1
    assert "Parse error:\nBadSyntax: broken" in prompt
    assert "OWL:\nnot run" in prompt
    assert "SHACL:\nnone" in prompt
    assert "Undefined terms:\nnone" in prompt
    assert "Previous output:\nnot turtle" in prompt
    assert prompt.rstrip().endswith("Corrected Turtle:")


def test_round0_is_a_cache_hit_and_a_clean_graph_stops(tmp_path) -> None:
    calls = {"n": 0}

    def respond(_request: LLMRequest) -> str:
        calls["n"] += 1
        return VALID

    client = make_client("mock", cache_dir=tmp_path / "cache", responder=respond)
    ontology = load_ontology(MINI)
    generate_individuals(
        ontology,
        str(MINI),
        client,
        target_class=TARGET,
        condition="A1",
        n_individuals=1,
        seed=0,
        model="mock-1",
        temperature=0.0,
        max_tokens=64,
        prompt_version="v3",
    )
    record = run_a2(
        ontology,
        MINI,
        client,
        target_class=TARGET,
        n_individuals=1,
        seed=0,
        model="mock-1",
        temperature=0.0,
        max_tokens=64,
        prompt_version="v3",
        max_repair_rounds=3,
        shapes=build_shapes(ontology).graph,
        baseline=check_consistency(MINI),
        check=True,
        require_cache_hit=True,
    )
    assert calls["n"] == 1
    assert record.round0_cached is True
    assert record.n_repair_rounds == 0
    assert record.stopped == "clean"
    assert record.rounds[0].feedback is None
    assert record.metrics.parse_ok is True
    assert record.metrics.inconsistent is False


def test_a_missing_cache_hit_is_rejected(tmp_path) -> None:
    client = make_client("mock", cache_dir=tmp_path / "cache", responder=lambda _request: VALID)
    ontology = load_ontology(MINI)
    try:
        run_a2(
            ontology,
            MINI,
            client,
            target_class=TARGET,
            n_individuals=1,
            seed=0,
            model="mock-1",
            temperature=0.0,
            max_tokens=64,
            prompt_version="v3",
            max_repair_rounds=3,
            shapes=build_shapes(ontology).graph,
            baseline=check_consistency(MINI),
            check=True,
            require_cache_hit=True,
        )
    except RepairError as exc:
        assert "cache hit" in str(exc)
    else:
        raise AssertionError("expected RepairError")


def test_repair_stops_when_the_next_completion_is_clean_and_caps_at_three() -> None:
    ontology = load_ontology(MINI)
    shapes = build_shapes(ontology).graph
    baseline = check_consistency(MINI)

    def repaired(request: LLMRequest) -> str:
        if "Previous output:" in request.prompt:
            return VALID
        return "this is not turtle"

    once = run_a2(
        ontology,
        MINI,
        make_client("mock", responder=repaired),
        target_class=TARGET,
        n_individuals=1,
        seed=0,
        model="mock-1",
        temperature=0.7,
        max_tokens=64,
        prompt_version="v3",
        max_repair_rounds=3,
        shapes=shapes,
        baseline=baseline,
        check=True,
        require_cache_hit=False,
    )
    assert once.n_repair_rounds == 1
    assert once.stopped == "clean"
    assert once.round0_cached is False
    assert once.rounds[0].feedback is not None
    assert "Parse error:" in once.rounds[0].feedback
    assert "this is not turtle" in once.rounds[0].feedback
    assert once.rounds[1].clean is True
    assert once.generation.temperature == 0.7

    always = run_a2(
        ontology,
        MINI,
        make_client("mock", responder=lambda _request: "nope"),
        target_class=TARGET,
        n_individuals=1,
        seed=1,
        model="mock-1",
        temperature=0.7,
        max_tokens=64,
        prompt_version="v3",
        max_repair_rounds=3,
        shapes=shapes,
        baseline=baseline,
        check=True,
        require_cache_hit=False,
    )
    assert always.n_repair_rounds == 3
    assert len(always.rounds) == 4
    assert always.stopped == "max_rounds"
    assert always.rounds[3].feedback is None
