"""Live check that Ollama's sampling seed repeats and that another seed changes.

The prompt is fixed and does not contain the seed. This is the check required
before a v3 run. It is not an experiment metric.
"""

from __future__ import annotations

from ollab.core.llm import LLMRequest, OllamaProvider, RawCompletion

PROBE_PROMPT = (
    "Write twelve distinct English words, one per line, with no numbering and no extra text."
)
PROBE_TEMPERATURE = 0.7
PROBE_MAX_TOKENS = 128
PROBE_SEED = 11
PROBE_OTHER = 29
PROBE_EXTRA = 43


def judge_seed(same_a: str, same_b: str, other: str, extra: str | None) -> str:
    """Return ``honored``, ``same-seed-differs``, or ``different-seed-matches``."""
    if same_a != same_b:
        return "same-seed-differs"
    if same_a != other:
        return "honored"
    if extra is not None and same_a != extra:
        return "honored"
    return "different-seed-matches"


def probe_seed(*, base_url: str, model: str, timeout_s: float = 180.0) -> dict[str, object]:
    """Call Ollama twice at one seed and once at another. A third seed breaks a tie."""
    if "Seed:" in PROBE_PROMPT or str(PROBE_SEED) in PROBE_PROMPT:
        raise RuntimeError("the seed probe prompt must not contain the seed")
    provider = OllamaProvider(base_url=base_url, timeout_s=timeout_s, max_attempts=1)
    try:
        first = _call(provider, model, PROBE_SEED)
        second = _call(provider, model, PROBE_SEED)
        other = _call(provider, model, PROBE_OTHER)
        extra: RawCompletion | None = None
        if first.text == second.text and first.text == other.text:
            extra = _call(provider, model, PROBE_EXTRA)
        calls = [first, second, other]
        seeds = [PROBE_SEED, PROBE_SEED, PROBE_OTHER]
        if extra is not None:
            calls.append(extra)
            seeds.append(PROBE_EXTRA)
        return {
            "done_reason": [item.done_reason for item in calls],
            "eval_count": [item.completion_tokens for item in calls],
            "max_tokens": PROBE_MAX_TOKENS,
            "model": model,
            "prompt_eval_count": [item.prompt_tokens for item in calls],
            "seeds": seeds,
            "status": judge_seed(
                first.text,
                second.text,
                other.text,
                None if extra is None else extra.text,
            ),
            "temperature": PROBE_TEMPERATURE,
            "texts": [item.text for item in calls],
        }
    finally:
        provider.close()


def _call(provider: OllamaProvider, model: str, seed: int) -> RawCompletion:
    return provider.complete(
        LLMRequest(
            model=model,
            prompt=PROBE_PROMPT,
            prompt_version="v3",
            temperature=PROBE_TEMPERATURE,
            seed=seed,
            max_tokens=PROBE_MAX_TOKENS,
        )
    )
