"""The summary keeps every condition, including a negative paired difference."""

from ollab.kgen.generate import GenerationRecord
from ollab.kgen.metrics import GraphMetrics
from ollab.kgen.records import RoundRecord, RunOutput, RunRecord
from ollab.kgen.summarize import summarize

_PIZZA = "http://example.org/mini#Pizza"
_NAMED = "http://example.org/mini#NamedPizza"


def _generation(
    condition: str,
    target_class: str,
    *,
    seed: int = 0,
    cached: bool = False,
    replicate_index: int = 0,
) -> GenerationRecord:
    return GenerationRecord(
        condition=condition,  # type: ignore[arg-type]
        seed=seed,
        target_class=target_class,
        provider="mock",
        model="mock-1",
        temperature=0.7,
        prompt_version="v3",
        replicate_index=replicate_index,
        prompt_hash="h",
        strict_parse_ok=True,
        lenient_parse_ok=True,
        prefix_completed_parse_ok=True,
        parse_error=None,
        n_requested=5,
        n_individuals=1,
        n_type_triples=1,
        n_property_triples=0,
        undefined_terms=(),
        turtle=None,
        raw_text="ind:1 a :Pizza .",
        prompt_eval_count=10,
        eval_count=4,
        done_reason="stop",
        truncated=False,
        cached=cached,
        llm_elapsed_s=0.5,
    )


def _metrics(*, parse_ok: bool, undefined: float | None) -> GraphMetrics:
    return GraphMetrics(
        parse_ok=parse_ok,
        truncated=False,
        n_undefined_terms=None if undefined is None else 0,
        n_vocabulary=None if undefined is None else 1,
        undefined_term_rate=undefined,
        n_class_as_individual=0,
        n_object_triples=0,
        class_as_individual_rate=None,
        n_individuals=0 if not parse_ok else 1,
        n_shacl_violations_generated=0 if parse_ok else None,
        shacl_violations_per_individual=0.0 if parse_ok else None,
        n_shacl_violations_other=0 if parse_ok else None,
        inconsistent=False if parse_ok else None,
        new_unsatisfiable_classes=() if parse_ok else None,
        new_unsatisfiable=False if parse_ok else None,
        coverage_count=1 if parse_ok else None,
        n_relevant=4 if parse_ok else None,
        coverage_rate=0.25 if parse_ok else None,
        n_distinct_descriptions=1 if parse_ok else None,
        duplicate_rate=0.0 if parse_ok else None,
    )


def _record(
    condition: str,
    target_class: str,
    *,
    parse_ok: bool,
    undefined: float | None,
    seed: int = 0,
    replicate_index: int = 0,
) -> RunRecord:
    generation = _generation(condition, target_class, seed=seed, replicate_index=replicate_index)
    metrics = _metrics(parse_ok=parse_ok, undefined=undefined)
    return RunRecord(
        condition=condition,  # type: ignore[arg-type]
        generation=generation,
        metrics=metrics,
        rounds=(
            RoundRecord(
                round_index=0,
                generation=generation,
                metrics=metrics,
                clean=parse_ok,
                validate_elapsed_s=0.25,
            ),
        ),
        n_repair_rounds=0,
        stopped="single",
    )


def test_summary_reports_the_negative_pair_and_excludes_null_rates() -> None:
    document = RunOutput(
        provider="mock",
        model="mock-1",
        ontology="mini.ttl",
        class_seed=0,
        target_classes=(_PIZZA, _NAMED),
        temperature=0,
        prompt_version="v3",
        n_individuals=5,
        max_tokens=4096,
        seeds=(0,),
        records=(
            _record("A0", _PIZZA, parse_ok=False, undefined=None),
            _record("A1", _PIZZA, parse_ok=True, undefined=0.0),
            _record("A0", _NAMED, parse_ok=True, undefined=0.5),
            _record("A1", _NAMED, parse_ok=True, undefined=0.0),
        ),
    )
    summary = summarize((document,), wall_clock_s=1.5)
    pooled = summary["pooled"]
    assert pooled["A0"]["parse_ok"]["estimate"] == 0.5
    assert pooled["A1"]["parse_ok"]["estimate"] == 1.0
    paired = summary["paired"]["A1-A0"]["parse_ok"]
    assert paired["estimate"] == 0.5
    assert paired["n"] == 2
    undefined = summary["paired"]["A1-A0"]["undefined_term_rate"]
    assert undefined["n"] == 1
    assert undefined["n_excluded"] == 1
    assert undefined["estimate"] == -0.5
    assert summary["n_units"] == 2
    assert summary["extrapolation"]["usd"] is None
    assert summary["extrapolation"]["phase_2b_units"] is None
    assert summary["per_unit"]["uncached_calls"] == 2.0
    assert summary["wall_clock_s"] == 1.5
    assert summary["method"]["unit"] == "target class IRI"


def test_two_generations_of_one_class_are_rejected() -> None:
    document = RunOutput(
        provider="mock",
        model="mock-1",
        ontology="mini.ttl",
        class_seed=0,
        target_classes=(_PIZZA,),
        seeds=(0, 1),
        records=(
            _record("A0", _PIZZA, parse_ok=True, undefined=0.0, seed=0),
            _record("A0", _PIZZA, parse_ok=True, undefined=0.0, seed=1),
        ),
    )
    try:
        summarize((document,), wall_clock_s=0)
    except ValueError as exc:
        assert "more than one generation" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_a_second_replicate_is_not_a_second_class() -> None:
    document = RunOutput(
        provider="mock",
        model="mock-1",
        ontology="mini.ttl",
        class_seed=0,
        target_classes=(_PIZZA,),
        seeds=(0,),
        replicates=(0, 1),
        records=(
            _record("A0", _PIZZA, parse_ok=True, undefined=0.0, replicate_index=0),
            _record("A0", _PIZZA, parse_ok=False, undefined=None, replicate_index=1),
        ),
    )
    summary = summarize((document,), wall_clock_s=1.0)
    assert summary["n_units"] == 1
    assert summary["pooled"]["A0"]["parse_ok"]["estimate"] == 1.0
    assert summary["per_unit"]["uncached_calls"] == 1.0
    assert summary["totals"]["uncached_calls"] == 2
    by_replicate = summary["by_replicate"]
    assert by_replicate["1"]["n_units"] == 1
    assert by_replicate["1"]["pooled"]["A0"]["parse_ok"]["estimate"] == 0.0
