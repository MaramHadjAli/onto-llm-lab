"""Run an A0/A1/A2 generation config and validate each Turtle graph."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rdflib import Graph

from ollab.core.llm import LLMClient, LLMRequest, make_client
from ollab.core.ontology import Ontology, load_ontology
from ollab.kgen.archive import CompletionArchive, fetch_ollama_version, new_run_id
from ollab.kgen.generate import PromptVersion, generate_individuals, select_classes
from ollab.kgen.records import RunOutput, RunRecord
from ollab.kgen.repair import assess_generation, run_a2
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_owl import check_consistency


class SafetyError(RuntimeError):
    """The run would write mock output into ``results/``."""


class KgenConfig(BaseModel):
    """Experiment settings read from a YAML file. Paths are relative to the cwd.

    Set ``ontology`` for one file, or ``ontologies`` for several. A2 is allowed
    only when A1 is listed earlier and validation is on, so round 0 can reuse
    the A1 completion and the repair loop can see HermiT and SHACL.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    ontology: Path | None = None
    ontologies: list[Path] = Field(default_factory=list)
    n_classes: int
    n_individuals: int
    seeds: list[int]
    replicates: list[int] = Field(default_factory=lambda: [0])
    class_seed: int = 0
    class_pool: int | None = None
    conditions: list[str]
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    num_ctx: int = Field(default=16384, gt=0)
    keep_alive: int | str | None = None
    warmup: bool = False
    concurrent_activity: str | None = None
    timeout_s: float = 120.0
    prompt_version: PromptVersion = "v2"
    max_repair_rounds: int = Field(default=3, ge=0, le=3)
    output: Path = Path("results/kgen_smoke.json")
    base_url: str | None = None
    check: bool = Field(default=True, alias="validate")

    @field_validator("keep_alive")
    @classmethod
    def _keep_alive(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("keep_alive must not be empty")
            return value
        if isinstance(value, bool) or (isinstance(value, int) and value < -1):
            raise ValueError("keep_alive must be -1 or a non-negative number of seconds")
        return value

    @model_validator(mode="after")
    def _protocol(self) -> Self:
        has_one = self.ontology is not None
        has_many = bool(self.ontologies)
        if has_one == has_many:
            raise ValueError("set exactly one of ontology or ontologies")
        allowed = {"A0", "A1", "A2"}
        unknown = [item for item in self.conditions if item not in allowed]
        if unknown:
            raise ValueError(f"unknown conditions: {', '.join(unknown)}")
        if "A2" in self.conditions:
            if "A1" not in self.conditions or self.conditions.index("A1") > self.conditions.index(
                "A2"
            ):
                raise ValueError("A1 must appear before A2 in conditions")
            if not self.check:
                raise ValueError("A2 requires validate: true")
        if not self.replicates:
            raise ValueError("replicates must not be empty")
        if any(item < 0 for item in self.replicates):
            raise ValueError("replicate_index must be non-negative")
        if len(set(self.replicates)) != len(self.replicates):
            raise ValueError("replicate indexes must be unique")
        if self.class_pool is not None and self.class_pool < self.n_classes:
            raise ValueError("class_pool must be at least n_classes")
        return self


def load_config(path: Path) -> KgenConfig:
    """Read a YAML config. Relative paths stay relative; the caller resolves them."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return KgenConfig.model_validate(payload)


def planned_classes(
    ontology: Ontology,
    *,
    n_classes: int,
    class_seed: int,
    class_pool: int | None = None,
) -> tuple[str, ...]:
    """Return the target classes for one ontology.

    ``class_pool`` draws that many classes and keeps the first ``n_classes``
    of the sorted draw. The noise study uses this so its 10 classes are a
    subset of the main run's 30, not a fresh sample of 10.
    """
    draw = n_classes if class_pool is None else class_pool
    chosen = select_classes(ontology, draw, class_seed)
    if class_pool is None:
        return chosen
    return chosen[:n_classes]


def is_results_path(path: Path, *, cwd: Path | None = None) -> bool:
    """True when ``path`` is inside the working directory's ``results`` folder."""
    root = (cwd or Path.cwd()).resolve()
    resolved = path if path.is_absolute() else root / path
    try:
        resolved.resolve().relative_to(root / "results")
    except ValueError:
        return False
    return True


def execute(
    config: KgenConfig,
    *,
    dry_run: bool = False,
    client: LLMClient | None = None,
    cwd: Path | None = None,
    archive: CompletionArchive | None = None,
    warmup_elapsed_s: float | None = None,
) -> RunOutput:
    """Generate individuals for every class, seed, and condition of one ontology.

    A mock provider may not write under ``results/`` unless ``dry_run`` is set.
    ``dry_run`` returns the document and does not create the output file.
    """
    if config.ontology is None:
        raise ValueError("execute() needs config.ontology; use execute_suite() for several")
    if config.provider == "mock" and not dry_run and is_results_path(config.output, cwd=cwd):
        raise SafetyError(
            "refusing to write mock output into results/; pass --dry-run or use a non-mock provider"
        )
    ontology_path = _resolve(config.ontology, cwd)
    ontology = load_ontology(ontology_path)
    classes = planned_classes(
        ontology,
        n_classes=config.n_classes,
        class_seed=config.class_seed,
        class_pool=config.class_pool,
    )
    llm = client or _make(config, cwd)
    shapes: Graph | None = None
    baseline = None
    baseline_elapsed = None
    if config.check:
        started = time.perf_counter()
        baseline = check_consistency(ontology_path)
        baseline_elapsed = time.perf_counter() - started
        shapes = build_shapes(ontology).graph
    records: list[RunRecord] = []

    def _save(generation, round_index: int) -> None:
        if archive is None:
            return
        archive.append(
            generation,
            ontology=str(ontology_path),
            round_index=round_index,
            num_ctx=config.num_ctx,
            max_tokens=config.max_tokens,
            class_seed=config.class_seed,
        )

    for replicate in config.replicates:
        for seed in config.seeds:
            for condition in config.conditions:
                for target in classes:
                    if condition == "A2":
                        records.append(
                            run_a2(
                                ontology,
                                ontology_path,
                                llm,
                                target_class=target,
                                n_individuals=config.n_individuals,
                                seed=seed,
                                model=config.model,
                                temperature=config.temperature,
                                max_tokens=config.max_tokens,
                                prompt_version=config.prompt_version,
                                max_repair_rounds=config.max_repair_rounds,
                                shapes=shapes,
                                baseline=baseline,
                                check=config.check,
                                require_cache_hit=True,
                                replicate_index=replicate,
                                on_completion=_save,
                            )
                        )
                        continue
                    generated = generate_individuals(
                        ontology,
                        str(ontology_path),
                        llm,
                        target_class=target,
                        condition=condition,  # type: ignore[arg-type]
                        n_individuals=config.n_individuals,
                        seed=seed,
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        prompt_version=config.prompt_version,
                        replicate_index=replicate,
                    )
                    _save(generated, 0)
                    metrics, consistency, conforms, violations, elapsed, clean = assess_generation(
                        ontology,
                        ontology_path,
                        generated,
                        shapes=shapes,
                        baseline=baseline,
                        check=config.check,
                    )
                    round_record = _round(
                        generated, metrics, consistency, conforms, violations, elapsed, clean
                    )
                    records.append(
                        RunRecord(
                            condition=condition,  # type: ignore[arg-type]
                            generation=generated,
                            metrics=metrics,
                            consistency=consistency,
                            shacl_conforms=conforms,
                            shacl_violations=violations or (),
                            rounds=(round_record,),
                            n_repair_rounds=0,
                            round0_cached=generated.cached,
                            stopped="single",
                        )
                    )
    return RunOutput(
        provider=config.provider,
        model=config.model,
        ontology=str(ontology_path),
        class_seed=config.class_seed,
        target_classes=classes,
        temperature=config.temperature,
        prompt_version=config.prompt_version,
        n_individuals=config.n_individuals,
        max_tokens=config.max_tokens,
        num_ctx=config.num_ctx,
        seeds=tuple(config.seeds),
        replicates=tuple(config.replicates),
        run_id=None if archive is None else archive.run_id,
        ollama_version=None if archive is None else archive.ollama_version,
        keep_alive=config.keep_alive,
        warmup=config.warmup,
        warmup_elapsed_s=warmup_elapsed_s,
        concurrent_activity=config.concurrent_activity,
        baseline_consistent=None if baseline is None else baseline.consistent,
        baseline_unsatisfiable_classes=() if baseline is None else baseline.unsatisfiable_classes,
        baseline_error=None if baseline is None else baseline.error,
        baseline_elapsed_s=baseline_elapsed,
        records=tuple(records),
    )


def execute_suite(
    config: KgenConfig,
    *,
    dry_run: bool = False,
    client: LLMClient | None = None,
    cwd: Path | None = None,
) -> tuple[tuple[RunOutput, ...], dict[str, object]]:
    """Run every ontology in the config with one shared LLM cache."""
    if config.provider == "mock" and not dry_run and is_results_path(config.output, cwd=cwd):
        raise SafetyError(
            "refusing to write mock output into results/; pass --dry-run or use a non-mock provider"
        )
    paths = tuple(config.ontologies) if config.ontologies else (config.ontology,)
    llm = client or _make(config, cwd)
    warmup_elapsed = _warmup(config, llm)
    archive = None
    if not dry_run:
        directory = _resolve(config.output, cwd)
        version = None
        if config.provider == "ollama":
            version = fetch_ollama_version(config.base_url or "http://127.0.0.1:11434")
        archive = CompletionArchive(
            directory / "raw_completions.jsonl",
            run_id=new_run_id(),
            ollama_version=version,
            concurrent_activity=config.concurrent_activity,
        )
    started = time.perf_counter()
    documents = []
    for path in paths:
        if path is None:
            continue
        one = config.model_copy(update={"ontology": path, "ontologies": []})
        documents.append(
            execute(
                one,
                dry_run=True,
                client=llm,
                cwd=cwd,
                archive=archive,
                warmup_elapsed_s=warmup_elapsed,
            )
        )
    wall_clock_s = time.perf_counter() - started
    from ollab.kgen.summarize import summarize

    return tuple(documents), summarize(documents, wall_clock_s=wall_clock_s)


def write_output(path: Path, document: RunOutput, *, cwd: Path | None = None) -> Path:
    """Write the run document as JSON. The parent directory is created."""
    destination = _resolve(path, cwd)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dump_document(document), encoding="utf-8")
    return destination


def write_suite(
    directory: Path,
    documents: tuple[RunOutput, ...],
    summary: dict[str, object],
    *,
    cwd: Path | None = None,
) -> Path:
    """Write one JSON file per ontology plus ``summary.json``."""
    destination = _resolve(directory, cwd)
    destination.mkdir(parents=True, exist_ok=True)
    for document in documents:
        stem = Path(document.ontology).stem
        (destination / f"{stem}.json").write_text(dump_document(document), encoding="utf-8")
    (destination / "summary.json").write_text(_dump_summary(summary), encoding="utf-8")
    return destination


def _round(generated, metrics, consistency, conforms, violations, elapsed, clean):
    from ollab.kgen.records import RoundRecord

    return RoundRecord(
        round_index=0,
        generation=generated,
        metrics=metrics,
        consistency=consistency,
        shacl_conforms=conforms,
        shacl_violations=violations or (),
        clean=clean,
        feedback=None,
        validate_elapsed_s=elapsed,
    )


def _warmup(config: KgenConfig, client: LLMClient) -> float | None:
    """Load the model with one short call. The text is not archived."""
    if not config.warmup:
        return None
    started = time.perf_counter()
    client.provider.complete(
        LLMRequest(
            model=config.model,
            prompt="Reply with the single word ready.",
            prompt_version=config.prompt_version,
            temperature=config.temperature,
            seed=0,
            max_tokens=8,
        )
    )
    return time.perf_counter() - started


def _make(config: KgenConfig, cwd: Path | None) -> LLMClient:
    return make_client(
        config.provider,
        base_url=config.base_url,
        cache_dir=_cache_dir(cwd),
        num_ctx=config.num_ctx,
        timeout_s=config.timeout_s,
        keep_alive=config.keep_alive,
    )


def _cache_dir(cwd: Path | None) -> Path:
    configured = os.environ.get("LLM_CACHE_DIR", "cache/llm")
    cache = Path(configured)
    if cache.is_absolute():
        return cache
    return (cwd or Path.cwd()) / cache


def _resolve(path: Path, cwd: Path | None) -> Path:
    if path.is_absolute():
        return path
    return (cwd or Path.cwd()) / path


def dump_document(document: RunOutput) -> str:
    """Return canonical JSON for stdout and for the results file."""
    return json.dumps(json.loads(document.model_dump_json()), indent=2, sort_keys=True) + "\n"


def _dump_summary(summary: dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"
