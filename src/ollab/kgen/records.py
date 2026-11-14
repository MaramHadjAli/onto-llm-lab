"""Generation records written by ``ollab kgen run``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ollab.kgen.generate import GenerationRecord
from ollab.kgen.metrics import GraphMetrics
from ollab.kgen.validate_owl import ConsistencyReport
from ollab.kgen.validate_shacl import ShaclViolation

Stopped = Literal["single", "clean", "max_rounds"]


class RoundRecord(BaseModel):
    """One completion inside a unit, with the metrics for that completion."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    generation: GenerationRecord
    metrics: GraphMetrics
    consistency: ConsistencyReport | None = None
    shacl_conforms: bool | None = None
    shacl_violations: tuple[ShaclViolation, ...] = ()
    clean: bool
    feedback: str | None = None
    validate_elapsed_s: float | None = None


class RunRecord(BaseModel):
    """One condition on one unit. ``metrics`` describes the scored graph.

    For A2 that graph is the last round. ``rounds`` keeps every round.
    ``n_repair_rounds`` counts repair calls and does not count round 0.
    """

    model_config = ConfigDict(frozen=True)

    condition: Literal["A0", "A1", "A2"]
    generation: GenerationRecord
    metrics: GraphMetrics
    consistency: ConsistencyReport | None = None
    shacl_conforms: bool | None = None
    shacl_violations: tuple[ShaclViolation, ...] = ()
    rounds: tuple[RoundRecord, ...] = ()
    n_repair_rounds: int = 0
    round0_cached: bool | None = None
    stopped: Stopped = "single"


class RunOutput(BaseModel):
    """The JSON document written for one ontology."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    ontology: str
    class_seed: int
    target_classes: tuple[str, ...]
    temperature: float | None = None
    prompt_version: str | None = None
    n_individuals: int | None = None
    max_tokens: int | None = None
    num_ctx: int | None = None
    seeds: tuple[int, ...] = ()
    replicates: tuple[int, ...] = ()
    run_id: str | None = None
    ollama_version: str | None = None
    baseline_consistent: bool | None = None
    baseline_unsatisfiable_classes: tuple[str, ...] = ()
    baseline_error: str | None = None
    baseline_elapsed_s: float | None = None
    records: tuple[RunRecord, ...] = Field(default_factory=tuple)
