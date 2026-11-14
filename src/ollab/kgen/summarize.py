"""Per-condition means, paired differences, and the Phase 2b rate sheet.

The formulas are the ones in ``docs/metrics.md``. This module does not drop
a negative estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

from ollab.core.stats import bootstrap_ci
from ollab.kgen.metrics import unit_truncated
from ollab.kgen.records import RunOutput, RunRecord

_RESAMPLES = 9999
_SEED = 0
_CONFIDENCE = 0.95
_PAIRS = (("A1", "A0"), ("A2", "A0"), ("A2", "A1"))
_METRICS = (
    "parse_ok",
    "truncated",
    "undefined_term_rate",
    "class_as_individual_rate",
    "shacl_violations_per_individual",
    "shacl_other_focus_count",
    "inconsistent",
    "new_unsatisfiable",
    "coverage_count",
    "coverage_rate",
    "duplicate_rate",
    "n_repair_rounds",
    "uncached_prompt_eval_count",
    "uncached_eval_count",
    "uncached_llm_elapsed_s",
)


def summarize(documents: Sequence[RunOutput], *, wall_clock_s: float) -> dict[str, object]:
    """Summarise one or more ontology documents from the same protocol."""
    table = _index(documents)
    keys = _primary_keys(table)
    pooled = {
        condition: _condition_summary(table, keys, condition) for condition in ("A0", "A1", "A2")
    }
    paired = {
        f"{left}-{right}": _paired_summary(table, keys, left, right) for left, right in _PAIRS
    }
    per_ontology: dict[str, object] = {}
    for ontology in sorted({key[0] for key in keys}):
        local = [key for key in keys if key[0] == ontology]
        per_ontology[ontology] = {
            condition: _condition_summary(table, local, condition)
            for condition in ("A0", "A1", "A2")
        }
    totals = _totals(documents)
    n_units = len(keys)
    primary_replicate = 0 if any(key[2] == 0 for key in table) else None
    summary: dict[str, object] = {
        "ci_note": (
            "Intervals are percentile bootstraps over target classes at "
            "replicate_index 0. Other replicates are not extra classes."
        ),
        "method": {
            "confidence": _CONFIDENCE,
            "n_resamples": _RESAMPLES,
            "seed": _SEED,
            "statistic": "mean",
            "unit": "target class IRI",
        },
        "n_units": n_units,
        "paired": paired,
        "per_ontology": per_ontology,
        "per_unit": _per_unit(
            _totals(documents, replicate=primary_replicate),
            n_units,
        ),
        "pooled": pooled,
        "protocol": _protocol(documents),
        "repair_example": _repair_example(documents),
        "totals": totals,
        "wall_clock_s": wall_clock_s,
        "extrapolation": {
            "formula": (
                "For G units on the same protocol, model, prompt version, and "
                "temperature: G times each per_unit rate. Add baseline elapsed "
                "seconds for each ontology in that grid. The repair rate and the "
                "class mix are those of this run. usd is null because no token "
                "price was supplied."
            ),
            "phase_2b_units": None,
            "usd": None,
        },
    }
    by_replicate = _by_replicate(table)
    if by_replicate is not None:
        summary["by_replicate"] = by_replicate
    return summary


def _index(
    documents: Sequence[RunOutput],
) -> dict[tuple[str, str, int], dict[str, RunRecord]]:
    table: dict[tuple[str, str, int], dict[str, RunRecord]] = {}
    for document in documents:
        for record in document.records:
            key = (
                document.ontology,
                record.generation.target_class,
                record.generation.replicate_index,
            )
            slot = table.setdefault(key, {})
            if record.condition in slot:
                raise ValueError(
                    "a target class has more than one generation for "
                    f"{record.condition}: {record.generation.target_class}"
                )
            slot[record.condition] = record
    return table


def _primary_keys(
    table: dict[tuple[str, str, int], dict[str, RunRecord]],
) -> list[tuple[str, str, int]]:
    """Replicate 0 is the published table. A run with no replicate 0 uses all rows."""
    zeros = [key for key in sorted(table) if key[2] == 0]
    if zeros:
        return zeros
    return sorted(table)


def _by_replicate(
    table: dict[tuple[str, str, int], dict[str, RunRecord]],
) -> dict[str, object] | None:
    indexes = sorted({key[2] for key in table})
    if len(indexes) < 2:
        return None
    reported: dict[str, object] = {}
    for replicate in indexes:
        keys = [key for key in sorted(table) if key[2] == replicate]
        reported[str(replicate)] = {
            "n_units": len(keys),
            "pooled": {
                condition: _condition_summary(table, keys, condition)
                for condition in ("A0", "A1", "A2")
            },
        }
    return reported


def _condition_summary(
    table: dict[tuple[str, str, int], dict[str, RunRecord]],
    keys: Sequence[tuple[str, str, int]],
    condition: str,
) -> dict[str, object]:
    records = [table[key][condition] for key in keys if condition in table[key]]
    values = {metric: [] for metric in _METRICS}
    for record in records:
        extracted = _unit_values(record)
        for metric in _METRICS:
            number = extracted[metric]
            if number is not None:
                values[metric].append(number)
    n_units = len(records)
    return {metric: _ci(values[metric], n_units) for metric in _METRICS}


def _paired_summary(
    table: dict[tuple[str, str, int], dict[str, RunRecord]],
    keys: Sequence[tuple[str, str, int]],
    left: str,
    right: str,
) -> dict[str, object]:
    both = [key for key in keys if left in table[key] and right in table[key]]
    values = {metric: [] for metric in _METRICS}
    for key in both:
        first = _unit_values(table[key][left])
        second = _unit_values(table[key][right])
        for metric in _METRICS:
            if first[metric] is None or second[metric] is None:
                continue
            values[metric].append(first[metric] - second[metric])
    n_units = len(both)
    return {metric: _ci(values[metric], n_units) for metric in _METRICS}


def _ci(values: list[float], n_units: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "confidence": _CONFIDENCE,
        "high": None,
        "low": None,
        "method": "percentile",
        "n": len(values),
        "n_excluded": n_units - len(values),
        "n_resamples": _RESAMPLES,
        "seed": _SEED,
    }
    if not values:
        summary["estimate"] = None
        return summary
    interval = bootstrap_ci(
        values,
        n_resamples=_RESAMPLES,
        confidence=_CONFIDENCE,
        seed=_SEED,
    )
    summary["estimate"] = interval.estimate
    summary["low"] = interval.low
    summary["high"] = interval.high
    return summary


def _unit_values(record: RunRecord) -> dict[str, float | None]:
    metrics = record.metrics
    flags = tuple(item.generation.truncated for item in record.rounds)
    return {
        "parse_ok": 1.0 if metrics.parse_ok else 0.0,
        "truncated": _as_float(unit_truncated(flags)),
        "undefined_term_rate": metrics.undefined_term_rate,
        "class_as_individual_rate": metrics.class_as_individual_rate,
        "shacl_violations_per_individual": metrics.shacl_violations_per_individual,
        "shacl_other_focus_count": _as_float(metrics.n_shacl_violations_other),
        "inconsistent": _as_float(metrics.inconsistent),
        "new_unsatisfiable": _as_float(metrics.new_unsatisfiable),
        "coverage_count": _as_float(metrics.coverage_count),
        "coverage_rate": metrics.coverage_rate,
        "duplicate_rate": metrics.duplicate_rate,
        "n_repair_rounds": float(record.n_repair_rounds),
        "uncached_prompt_eval_count": _uncached_sum(record, "prompt_eval_count"),
        "uncached_eval_count": _uncached_sum(record, "eval_count"),
        "uncached_llm_elapsed_s": _uncached_sum(record, "llm_elapsed_s"),
    }


def _as_float(value: bool | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _uncached_sum(record: RunRecord, field: str) -> float | None:
    total = 0.0
    saw = False
    for item in record.rounds:
        if item.generation.cached:
            continue
        value = getattr(item.generation, field)
        if value is None:
            return None
        total += float(value)
        saw = True
    if not saw:
        return 0.0
    return total


def _totals(documents: Sequence[RunOutput], *, replicate: int | None = None) -> dict[str, object]:
    uncached_calls = 0
    cached_calls = 0
    prompt = 0
    completion = 0
    missing_prompt = 0
    missing_completion = 0
    llm_s = 0.0
    validate_s = 0.0
    truncated_completions = 0
    a2_round0 = 0
    a2_round0_cached = 0
    for document in documents:
        for record in document.records:
            if replicate is not None and record.generation.replicate_index != replicate:
                continue
            if record.condition == "A2" and record.rounds:
                a2_round0 += 1
                if record.rounds[0].generation.cached:
                    a2_round0_cached += 1
            for item in record.rounds:
                if item.generation.truncated is True:
                    truncated_completions += 1
                validate_s += item.validate_elapsed_s or 0.0
                if item.generation.cached:
                    cached_calls += 1
                    continue
                uncached_calls += 1
                if item.generation.prompt_eval_count is None:
                    missing_prompt += 1
                else:
                    prompt += item.generation.prompt_eval_count
                if item.generation.eval_count is None:
                    missing_completion += 1
                else:
                    completion += item.generation.eval_count
                if item.generation.llm_elapsed_s is not None:
                    llm_s += item.generation.llm_elapsed_s
    return {
        "a2_round0_cached": a2_round0_cached,
        "a2_round0_calls": a2_round0,
        "cached_calls": cached_calls,
        "truncated_completions": truncated_completions,
        "uncached_calls": uncached_calls,
        "uncached_calls_missing_completion_tokens": missing_completion,
        "uncached_calls_missing_prompt_tokens": missing_prompt,
        "uncached_eval_count": completion,
        "uncached_llm_elapsed_s": llm_s,
        "uncached_prompt_eval_count": prompt,
        "validate_elapsed_s": validate_s,
        "baselines": [
            {
                "consistent": document.baseline_consistent,
                "elapsed_s": document.baseline_elapsed_s,
                "error": document.baseline_error,
                "ontology": document.ontology,
                "unsatisfiable_classes": list(document.baseline_unsatisfiable_classes),
            }
            for document in documents
        ],
    }


def _per_unit(totals: dict[str, object], n_units: int) -> dict[str, float | None]:
    if n_units == 0:
        return {
            "uncached_calls": None,
            "uncached_completion_tokens": None,
            "uncached_llm_elapsed_s": None,
            "uncached_prompt_tokens": None,
            "validate_elapsed_s": None,
        }
    missing_prompt = int(totals["uncached_calls_missing_prompt_tokens"])
    missing_completion = int(totals["uncached_calls_missing_completion_tokens"])
    return {
        "uncached_calls": int(totals["uncached_calls"]) / n_units,
        "uncached_completion_tokens": (
            None if missing_completion else int(totals["uncached_eval_count"]) / n_units
        ),
        "uncached_llm_elapsed_s": float(totals["uncached_llm_elapsed_s"]) / n_units,
        "uncached_prompt_tokens": (
            None if missing_prompt else int(totals["uncached_prompt_eval_count"]) / n_units
        ),
        "validate_elapsed_s": float(totals["validate_elapsed_s"]) / n_units,
    }


def _protocol(documents: Sequence[RunOutput]) -> dict[str, object]:
    if not documents:
        return {}
    first = documents[0]
    return {
        "concurrent_activity": first.concurrent_activity,
        "keep_alive": first.keep_alive,
        "warmup": first.warmup,
        "max_tokens": first.max_tokens,
        "model": first.model,
        "n_individuals": first.n_individuals,
        "num_ctx": first.num_ctx,
        "ontologies": [document.ontology for document in documents],
        "prompt_version": first.prompt_version,
        "provider": first.provider,
        "replicates": list(first.replicates) if first.replicates else [0],
        "seeds": list(first.seeds),
        "target_classes": {
            document.ontology: list(document.target_classes) for document in documents
        },
        "temperature": first.temperature,
    }


def _repair_example(documents: Sequence[RunOutput]) -> dict[str, object] | None:
    """The A2 trace with the most repair rounds. Ties break by ontology, class, seed."""
    candidates: list[tuple[str, RunRecord]] = []
    for document in documents:
        for record in document.records:
            if record.condition == "A2":
                candidates.append((document.ontology, record))
    if not candidates:
        return None
    ontology, record = min(
        candidates,
        key=lambda item: (
            -item[1].n_repair_rounds,
            item[0],
            item[1].generation.target_class,
            item[1].generation.seed,
        ),
    )
    return {
        "n_repair_rounds": record.n_repair_rounds,
        "ontology": ontology,
        "rounds": [_round_dump(item) for item in record.rounds],
        "seed": record.generation.seed,
        "stopped": record.stopped,
        "target_class": record.generation.target_class,
    }


def _round_dump(item) -> dict[str, object]:
    consistency = None
    if item.consistency is not None:
        consistency = item.consistency.model_dump(mode="json")
    return {
        "cached": item.generation.cached,
        "clean": item.clean,
        "consistency": consistency,
        "done_reason": item.generation.done_reason,
        "eval_count": item.generation.eval_count,
        "feedback": item.feedback,
        "llm_elapsed_s": item.generation.llm_elapsed_s,
        "metrics": item.metrics.model_dump(mode="json"),
        "parse_error": item.generation.parse_error,
        "parse_ok": item.metrics.parse_ok,
        "prompt_eval_count": item.generation.prompt_eval_count,
        "raw_text": item.generation.raw_text,
        "round_index": item.round_index,
        "shacl_violations": [row.model_dump(mode="json") for row in item.shacl_violations],
        "truncated": item.generation.truncated,
        "validate_elapsed_s": item.validate_elapsed_s,
    }
