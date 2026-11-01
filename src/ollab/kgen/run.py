"""Run an A0/A1 generation config and optionally validate each Turtle graph."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ollab.core.llm import LLMClient, make_client
from ollab.core.ontology import load_ontology
from ollab.kgen.generate import Condition, GenerationRecord, generate_individuals, select_classes
from ollab.kgen.shapes import build_shapes
from ollab.kgen.validate_owl import ConsistencyReport, check_consistency
from ollab.kgen.validate_shacl import ShaclViolation, validate_shacl


class SafetyError(RuntimeError):
    """The run would write mock output into ``results/``."""


class KgenConfig(BaseModel):
    """Experiment settings read from a YAML file. Paths are relative to the cwd."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    ontology: Path
    n_classes: int
    n_individuals: int
    seeds: list[int]
    class_seed: int = 0
    conditions: list[Condition]
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    output: Path = Path("results/kgen_smoke.json")
    base_url: str | None = None
    check: bool = Field(default=True, alias="validate")


class RunRecord(BaseModel):
    """A generation record plus the validation reports for that Turtle graph."""

    model_config = ConfigDict(frozen=True)

    generation: GenerationRecord
    consistency: ConsistencyReport | None = None
    shacl_conforms: bool | None = None
    shacl_violations: tuple[ShaclViolation, ...] = ()


class RunOutput(BaseModel):
    """The JSON document written by ``ollab kgen run``."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    ontology: str
    class_seed: int
    target_classes: tuple[str, ...]
    records: tuple[RunRecord, ...] = Field(default_factory=tuple)


def load_config(path: Path) -> KgenConfig:
    """Read a YAML config. Relative paths stay relative; the caller resolves them."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return KgenConfig.model_validate(payload)


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
) -> RunOutput:
    """Generate individuals for every class, seed, and condition.

    A mock provider may not write under ``results/`` unless ``dry_run`` is set.
    ``dry_run`` returns the document and does not create the output file.
    """
    if config.provider == "mock" and not dry_run and is_results_path(config.output, cwd=cwd):
        raise SafetyError(
            "refusing to write mock output into results/; pass --dry-run or use a non-mock provider"
        )
    ontology_path = _resolve(config.ontology, cwd)
    ontology = load_ontology(ontology_path)
    classes = select_classes(ontology, config.n_classes, config.class_seed)
    llm = client or make_client(
        config.provider,
        base_url=config.base_url,
        cache_dir=_cache_dir(cwd),
    )
    shapes = build_shapes(ontology).graph if config.check else None
    records: list[RunRecord] = []
    for seed in config.seeds:
        for condition in config.conditions:
            for target in classes:
                generated = generate_individuals(
                    ontology,
                    str(ontology_path),
                    llm,
                    target_class=target,
                    condition=condition,
                    n_individuals=config.n_individuals,
                    seed=seed,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                consistency = None
                conforms: bool | None = None
                violations: tuple[ShaclViolation, ...] = ()
                if config.check and generated.parse_ok and generated.turtle:
                    consistency = check_consistency(ontology_path, generated.turtle)
                    if shapes is not None:
                        conforms, violations = validate_shacl(
                            ontology_path, generated.turtle, shapes
                        )
                records.append(
                    RunRecord(
                        generation=generated,
                        consistency=consistency,
                        shacl_conforms=conforms,
                        shacl_violations=violations,
                    )
                )
    return RunOutput(
        provider=config.provider,
        model=config.model,
        ontology=str(ontology_path),
        class_seed=config.class_seed,
        target_classes=classes,
        records=tuple(records),
    )


def write_output(path: Path, document: RunOutput, *, cwd: Path | None = None) -> Path:
    """Write the run document as JSON. The parent directory is created."""
    destination = _resolve(path, cwd)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dump_document(document), encoding="utf-8")
    return destination


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
