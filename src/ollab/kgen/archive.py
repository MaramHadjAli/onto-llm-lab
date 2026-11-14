"""Append-only archive of raw completions for one kgen run.

The file is the source of truth for any reported number. A new run truncates
it, then each completion appends one JSON line so a crash keeps the calls
that finished. ``seed``, ``provider``, ``cached``, ``n_requested``, and
``class_seed`` are stored with the requested fields because offline rebuild
needs the prefix namespace, the provider label, the cache-hit flag, and the
requested individual count. ``replicate_index`` is stored here and is not
part of the prompt.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ollab.kgen.generate import GenerationRecord


def new_run_id() -> str:
    """UTC timestamp that identifies one suite. It is not a sampling seed."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def fetch_ollama_version(base_url: str) -> str | None:
    """Return the Ollama server version, or null when the request fails.

    A failed lookup is stored as null. This function does not invent a version.
    """
    url = base_url.rstrip("/") + "/api/version"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, str) and version:
        return version
    return None


class CompletionArchive:
    """One JSON line per completion, including each A2 round."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        ollama_version: str | None,
        concurrent_activity: str | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.ollama_version = ollama_version
        self.concurrent_activity = concurrent_activity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(
        self,
        generation: GenerationRecord,
        *,
        ontology: str,
        round_index: int,
        num_ctx: int,
        max_tokens: int,
        class_seed: int,
    ) -> None:
        """Append the raw completion. The line is flushed before returning."""
        row = {
            "cached": generation.cached,
            "class_seed": class_seed,
            "concurrent_activity": self.concurrent_activity,
            "condition": generation.condition,
            "done_reason": generation.done_reason,
            "eval_count": generation.eval_count,
            "max_tokens": max_tokens,
            "model": generation.model,
            "n_requested": generation.n_requested,
            "num_ctx": num_ctx,
            "ollama_version": self.ollama_version,
            "ontology": ontology,
            "prompt_eval_count": generation.prompt_eval_count,
            "prompt_hash": generation.prompt_hash,
            "prompt_version": generation.prompt_version,
            "provider": generation.provider,
            "raw_text": generation.raw_text,
            "replicate_index": generation.replicate_index,
            "round": round_index,
            "run_id": self.run_id,
            "seed": generation.seed,
            "target_class": generation.target_class,
            "temperature": generation.temperature,
            "wall_clock_s": generation.llm_elapsed_s,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
