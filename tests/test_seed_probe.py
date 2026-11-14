"""The seed-probe decision. The live Ollama call is not part of this test."""

from ollab.kgen.seed_probe import PROBE_PROMPT, judge_seed


def test_probe_prompt_does_not_contain_a_seed() -> None:
    assert "Seed:" not in PROBE_PROMPT
    assert "11" not in PROBE_PROMPT


def test_judge_seed_requires_a_repeat_and_a_difference() -> None:
    assert judge_seed("a", "a", "b", None) == "honored"
    assert judge_seed("a", "a", "a", "b") == "honored"
    assert judge_seed("a", "c", "b", None) == "same-seed-differs"
    assert judge_seed("a", "a", "a", "a") == "different-seed-matches"
    assert judge_seed("a", "a", "a", None) == "different-seed-matches"
