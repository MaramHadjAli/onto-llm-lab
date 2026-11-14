"""A2 repair loop. Round 0 is the A1 completion. At most three repairs follow.

Feedback text is the block in ``docs/prompt_versions.md``. A round is clean
under the rule in ``docs/metrics.md``.
"""

from __future__ import annotations

import time
from pathlib import Path

from rdflib import Graph

from ollab.core.llm import LLMClient
from ollab.core.ontology import Ontology
from ollab.kgen.generate import (
    PromptVersion,
    build_prompt,
    complete_prompt,
    generate_individuals,
)
from ollab.kgen.metrics import GraphMetrics, score_graph
from ollab.kgen.records import RoundRecord, RunRecord
from ollab.kgen.validate_owl import ConsistencyReport, check_consistency
from ollab.kgen.validate_shacl import ShaclViolation, validate_shacl

_INTRO = (
    "The previous answer was rejected. Return the full corrected Turtle triples only. "
    "Do not write @prefix lines and do not use a markdown fence: the prefixes above "
    "are already declared."
)
_SHACL_CAP = 10


class RepairError(RuntimeError):
    """A2 round 0 did not reuse the cached A1 completion."""


def shacl_feedback(messages: list[str], *, cap: int = _SHACL_CAP) -> str:
    """Deduplicated SHACL messages, sorted, capped, with an omission line."""
    unique = sorted(set(messages))
    if not unique:
        return "none"
    shown = unique[:cap]
    lines = [f"- {message}" for message in shown]
    omitted = len(unique) - len(shown)
    if omitted:
        lines.append(f"{omitted} further distinct messages omitted.")
    return "\n".join(lines)


def owl_feedback(validation_ran: bool, consistency: ConsistencyReport | None) -> str:
    """OWL status line for the repair prompt."""
    if not validation_ran or consistency is None:
        return "not run"
    if consistency.error:
        return consistency.error
    if consistency.consistent is False:
        return "inconsistent"
    if consistency.consistent is True:
        return "none"
    return "not run"


def render_feedback(
    *,
    parse_error: str | None,
    validation_ran: bool,
    consistency: ConsistencyReport | None,
    violations: tuple[ShaclViolation, ...] | None,
    undefined: tuple[str, ...],
    raw_text: str,
) -> str:
    """The feedback block appended to the A1 prompt."""
    terms = "none" if not undefined else "\n".join(f"- {iri}" for iri in undefined)
    messages = [] if violations is None else [row.message for row in violations]
    lines = [
        _INTRO,
        "",
        "Parse error:",
        parse_error or "none",
        "",
        "OWL:",
        owl_feedback(validation_ran, consistency),
        "",
        "SHACL:",
        shacl_feedback(messages),
        "",
        "Undefined terms:",
        terms,
        "",
        "Previous output:",
        raw_text,
        "",
        "Corrected Turtle:",
        "",
    ]
    return "\n".join(lines)


def repair_prompt(a1_prompt: str, feedback: str) -> str:
    """A1 prompt, a blank line, then the feedback block."""
    base = a1_prompt if a1_prompt.endswith("\n") else a1_prompt + "\n"
    return base + "\n" + feedback


def is_clean(
    *,
    parse_ok: bool,
    undefined: tuple[str, ...],
    consistency: ConsistencyReport | None,
    n_violations: int | None,
) -> bool:
    """True when the round has nothing the repair prompt would report."""
    return bool(
        parse_ok
        and not undefined
        and consistency is not None
        and consistency.consistent is True
        and n_violations == 0
    )


def assess_generation(
    ontology: Ontology,
    ontology_path: Path,
    generation,
    *,
    shapes: Graph | None,
    baseline: ConsistencyReport | None,
    check: bool,
) -> tuple[
    GraphMetrics,
    ConsistencyReport | None,
    bool | None,
    tuple[ShaclViolation, ...] | None,
    float,
    bool,
]:
    """Validate one completion and score it. Returns elapsed validation seconds."""
    parsed = _parsed_graph(generation.turtle) if generation.prefix_completed_parse_ok else None
    consistency: ConsistencyReport | None = None
    conforms: bool | None = None
    violations: tuple[ShaclViolation, ...] | None = None
    elapsed = 0.0
    if check and parsed is not None and generation.turtle and shapes is not None:
        started = time.perf_counter()
        consistency = check_consistency(ontology_path, generation.turtle)
        conforms, found = validate_shacl(ontology_path, generation.turtle, shapes)
        elapsed = time.perf_counter() - started
        violations = found
    metrics = score_graph(
        ontology,
        generation.target_class,
        parsed=parsed,
        parse_ok=parsed is not None,
        truncated=generation.truncated,
        consistency=consistency,
        violations=violations,
        baseline=baseline,
    )
    clean = is_clean(
        parse_ok=generation.prefix_completed_parse_ok,
        undefined=generation.undefined_terms,
        consistency=consistency,
        n_violations=None if violations is None else len(violations),
    )
    return metrics, consistency, conforms, violations, elapsed, clean


def run_a2(
    ontology: Ontology,
    ontology_path: Path,
    client: LLMClient,
    *,
    target_class: str,
    n_individuals: int,
    seed: int,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt_version: PromptVersion,
    max_repair_rounds: int,
    shapes: Graph | None,
    baseline: ConsistencyReport | None,
    check: bool,
    require_cache_hit: bool,
) -> RunRecord:
    """Run A2 for one unit. Round 0 calls the A1 request so the cache can hit."""
    if max_repair_rounds < 0 or max_repair_rounds > 3:
        raise ValueError("max_repair_rounds must be from 0 to 3")
    a1_prompt = build_prompt(
        ontology,
        target_class,
        condition="A1",
        n_individuals=n_individuals,
        seed=seed,
        prompt_version=prompt_version,
    )
    initial = generate_individuals(
        ontology,
        str(ontology_path),
        client,
        target_class=target_class,
        condition="A1",
        n_individuals=n_individuals,
        seed=seed,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_version=prompt_version,
    )
    if require_cache_hit and not initial.cached:
        raise RepairError(
            "A2 round 0 was not a cache hit for "
            f"{target_class} seed {seed}. The A1 request has to be completed first."
        )
    current = initial.model_copy(update={"condition": "A2"})
    rounds: list[RoundRecord] = []
    index = 0
    while True:
        metrics, consistency, conforms, violations, elapsed, clean = assess_generation(
            ontology,
            ontology_path,
            current,
            shapes=shapes,
            baseline=baseline,
            check=check,
        )
        will_repair = (not clean) and index < max_repair_rounds
        feedback = None
        if will_repair:
            feedback = render_feedback(
                parse_error=current.parse_error,
                validation_ran=violations is not None,
                consistency=consistency,
                violations=violations,
                undefined=current.undefined_terms if current.prefix_completed_parse_ok else (),
                raw_text=current.raw_text,
            )
        rounds.append(
            RoundRecord(
                round_index=index,
                generation=current,
                metrics=metrics,
                consistency=consistency,
                shacl_conforms=conforms,
                shacl_violations=violations or (),
                clean=clean,
                feedback=feedback,
                validate_elapsed_s=elapsed,
            )
        )
        if not will_repair:
            break
        current = complete_prompt(
            ontology,
            client,
            prompt=repair_prompt(a1_prompt, feedback or ""),
            condition="A2",
            target_class=target_class,
            n_individuals=n_individuals,
            seed=seed,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )
        index += 1
    last = rounds[-1]
    return RunRecord(
        condition="A2",
        generation=last.generation,
        metrics=last.metrics,
        consistency=last.consistency,
        shacl_conforms=last.shacl_conforms,
        shacl_violations=last.shacl_violations,
        rounds=tuple(rounds),
        n_repair_rounds=len(rounds) - 1,
        round0_cached=initial.cached,
        stopped="clean" if last.clean else "max_rounds",
    )


def _parsed_graph(turtle: str | None) -> Graph | None:
    if not turtle:
        return None
    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    return graph
