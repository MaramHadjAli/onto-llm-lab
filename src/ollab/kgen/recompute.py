"""Rebuild parse, validation, and summary tables from an archive.

This module loads local ontology files and runs HermiT and SHACL. It does
not construct an LLM client and it does not open a network connection.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from ollab.core.ontology import load_ontology
from ollab.kgen.generate import PromptVersion, record_from_completion
from ollab.kgen.records import RoundRecord, RunOutput, RunRecord
from ollab.kgen.repair import assess_generation
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_owl import check_consistency


def recompute_archive(path: Path) -> tuple[tuple[RunOutput, ...], dict[str, object]]:
    """Rebuild every ontology document and the summary from ``path``."""
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"archive is empty: {path}")
    run_ids = {str(row["run_id"]) for row in rows}
    if len(run_ids) != 1:
        raise ValueError(f"archive has {len(run_ids)} run ids; refusing to mix them")
    grouped: dict[str, list[dict[str, object]]] = {}
    order: list[str] = []
    for row in rows:
        ontology = str(row["ontology"])
        if ontology not in grouped:
            order.append(ontology)
            grouped[ontology] = []
        grouped[ontology].append(row)
    started = time.perf_counter()
    documents = tuple(_document(ontology, grouped[ontology]) for ontology in order)
    wall_clock_s = time.perf_counter() - started
    from ollab.kgen.summarize import summarize

    return documents, summarize(documents, wall_clock_s=wall_clock_s)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(loaded)
    return rows


def _document(ontology_path: str, rows: list[dict[str, object]]) -> RunOutput:
    path = Path(ontology_path)
    ontology = load_ontology(path)
    baseline_started = time.perf_counter()
    baseline = check_consistency(path)
    baseline_elapsed = time.perf_counter() - baseline_started
    shapes = build_shapes(ontology).graph
    classes: list[str] = []
    replicates: list[int] = []
    seeds: list[int] = []
    conditions: list[str] = []
    for row in rows:
        _remember(classes, str(row["target_class"]))
        _remember_int(replicates, row["replicate_index"])
        _remember_int(seeds, row["seed"])
        _remember(conditions, str(row["condition"]))
    groups: dict[tuple[int, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["replicate_index"]),
            int(row["seed"]),
            str(row["condition"]),
            str(row["target_class"]),
        )
        groups[key].append(row)
    records: list[RunRecord] = []
    for replicate in replicates:
        for seed in seeds:
            for condition in conditions:
                for target in classes:
                    bucket = groups.get((replicate, seed, condition, target))
                    if not bucket:
                        continue
                    records.append(
                        _run_record(
                            ontology,
                            path,
                            bucket,
                            shapes=shapes,
                            baseline=baseline,
                        )
                    )
    first = rows[0]
    return RunOutput(
        provider=str(first["provider"]),
        model=str(first["model"]),
        ontology=ontology_path,
        class_seed=int(first["class_seed"]),
        target_classes=tuple(classes),
        temperature=float(first["temperature"]),
        prompt_version=str(first["prompt_version"]),
        n_individuals=int(first["n_requested"]),
        max_tokens=int(first["max_tokens"]),
        num_ctx=int(first["num_ctx"]),
        seeds=tuple(seeds),
        replicates=tuple(replicates),
        run_id=str(first["run_id"]),
        ollama_version=None if first["ollama_version"] is None else str(first["ollama_version"]),
        baseline_consistent=baseline.consistent,
        baseline_unsatisfiable_classes=baseline.unsatisfiable_classes,
        baseline_error=baseline.error,
        baseline_elapsed_s=baseline_elapsed,
        records=tuple(records),
    )


def _run_record(ontology, ontology_path: Path, rows: list[dict[str, object]], *, shapes, baseline):
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    if ordered[0]["condition"] != "A2" and len(ordered) != 1:
        raise ValueError(
            f"more than one completion for {ordered[0]['condition']} {ordered[0]['target_class']}"
        )
    rounds: list[RoundRecord] = []
    for row in ordered:
        generation = record_from_completion(
            ontology,
            raw_text=str(row["raw_text"]),
            condition=row["condition"],  # type: ignore[arg-type]
            target_class=str(row["target_class"]),
            n_individuals=int(row["n_requested"]),
            seed=int(row["seed"]),
            model=str(row["model"]),
            provider=str(row["provider"]),
            temperature=float(row["temperature"]),
            prompt_version=_version(row["prompt_version"]),
            prompt_hash=str(row["prompt_hash"]),
            replicate_index=int(row["replicate_index"]),
            prompt_eval_count=_optional_int(row["prompt_eval_count"]),
            eval_count=_optional_int(row["eval_count"]),
            done_reason=None if row["done_reason"] is None else str(row["done_reason"]),
            cached=bool(row["cached"]),
            llm_elapsed_s=None if row["wall_clock_s"] is None else float(row["wall_clock_s"]),
        )
        metrics, consistency, conforms, violations, elapsed, clean = assess_generation(
            ontology,
            ontology_path,
            generation,
            shapes=shapes,
            baseline=baseline,
            check=True,
        )
        rounds.append(
            RoundRecord(
                round_index=int(row["round"]),
                generation=generation,
                metrics=metrics,
                consistency=consistency,
                shacl_conforms=conforms,
                shacl_violations=violations or (),
                clean=clean,
                feedback=None,
                validate_elapsed_s=elapsed,
            )
        )
    last = rounds[-1]
    condition = last.generation.condition
    if condition == "A2":
        stopped = "clean" if last.clean else "max_rounds"
        n_repair = len(rounds) - 1
    else:
        stopped = "single"
        n_repair = 0
    return RunRecord(
        condition=condition,  # type: ignore[arg-type]
        generation=last.generation,
        metrics=last.metrics,
        consistency=last.consistency,
        shacl_conforms=last.shacl_conforms,
        shacl_violations=last.shacl_violations,
        rounds=tuple(rounds),
        n_repair_rounds=n_repair,
        round0_cached=rounds[0].generation.cached,
        stopped=stopped,
    )


def _version(value: object) -> PromptVersion:
    if value not in ("v1", "v2", "v3"):
        raise ValueError(f"prompt_version must be 'v1', 'v2', or 'v3', got {value!r}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _remember(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _remember_int(items: list[int], value: object) -> None:
    number = int(value)
    if number not in items:
        items.append(number)
