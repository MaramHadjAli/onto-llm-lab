"""LLM client cache, mock, retries, and usage log. No network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ollab.core.llm import (
    LLMClientError,
    LLMRequest,
    TokenPrices,
    cache_key,
    default_mock_text,
    make_client,
)


def _request(**kwargs: object) -> LLMRequest:
    payload = {"model": "mock-1", "prompt": "hello world", "temperature": 0.0, "seed": 1}
    payload.update(kwargs)
    return LLMRequest(**payload)  # type: ignore[arg-type]


class ScriptedTransport(httpx.BaseTransport):
    def __init__(self, events: list[httpx.Response | Exception]) -> None:
        self.events = list(events)
        self.calls = 0
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def _openai_ok(text: str = "hello", prompt: int = 3, completion: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-test",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def test_mock_is_deterministic_and_records_seed() -> None:
    client = make_client("mock")
    first = client.complete(_request())
    second = client.complete(_request())
    assert first.text == second.text == default_mock_text(_request())
    assert first.seed == 1
    assert first.temperature == 0.0
    assert first.usage_source == "whitespace_estimate"
    assert first.prompt_tokens == 2
    assert first.cached is False
    assert first.cost_usd is None


def test_mock_does_not_construct_an_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("mock client opened an HTTP client")

    monkeypatch.setattr(httpx.Client, "__init__", explode)
    client = make_client("mock")
    response = client.complete(_request(prompt="offline"))
    assert response.text.startswith("mock-response")


def test_cache_replays_and_logs_without_storing_the_secret(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    usage = tmp_path / "usage.jsonl"
    calls = {"n": 0}

    def responder(request: LLMRequest) -> str:
        calls["n"] += 1
        return f"answer {request.prompt}"

    prices = TokenPrices(prompt_usd_per_million=2.0, completion_usd_per_million=4.0)
    client = make_client(
        "mock",
        cache_dir=cache_dir,
        usage_log=usage,
        prices=prices,
        responder=responder,
    )
    request = _request(prompt="one two three four")
    first = client.complete(request)
    second = client.complete(request)
    assert calls["n"] == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == "answer one two three four"
    # "one two three four" is 4 prompt tokens. The mock answer is 5 tokens.
    assert first.prompt_tokens == 4
    assert first.completion_tokens == 5
    assert first.cost_usd == pytest.approx((4 * 2.0 + 5 * 4.0) / 1_000_000)

    other_prices = TokenPrices(prompt_usd_per_million=9.0, completion_usd_per_million=9.0)
    replay = make_client("mock", cache_dir=cache_dir, prices=other_prices, responder=responder)
    third = replay.complete(request)
    assert third.cached is True
    assert third.cost_usd == first.cost_usd
    assert calls["n"] == 1

    raw_cache = "\n".join(path.read_text(encoding="utf-8") for path in cache_dir.rglob("*.json"))
    assert "one two three four" in raw_cache
    lines = [json.loads(line) for line in usage.read_text(encoding="utf-8").splitlines()]
    assert [line["cached"] for line in lines] == [False, True]
    assert "one two three four" not in usage.read_text(encoding="utf-8")
    assert lines[0]["seed"] == 1
    assert lines[0]["usage_source"] == "whitespace_estimate"


def test_cache_key_changes_with_seed_temperature_and_prompt() -> None:
    base = _request()
    original = cache_key("mock", "mock://local", base)
    assert cache_key("mock", "mock://local", _request(seed=2)) != original
    assert cache_key("mock", "mock://local", _request(temperature=0.2)) != original
    assert cache_key("mock", "mock://local", _request(prompt="other")) != original
    assert cache_key("openai", "mock://local", base) != original


def test_openai_request_shape_retries_and_redaction(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    class Recording(ScriptedTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content.decode("utf-8")))
            return super().handle_request(request)

    transport = Recording(
        [
            httpx.Response(503, json={"error": "busy"}),
            httpx.ConnectError("down"),
            _openai_ok(),
        ]
    )
    client = make_client(
        "openai",
        api_key="sk-secret",
        base_url="https://example.invalid/v1",
        transport=transport,
        cache_dir=tmp_path / "cache",
        max_attempts=3,
        backoff_s=0,
    )
    response = client.complete(_request(model="gpt-test", system="be brief", seed=4, max_tokens=16))
    assert transport.calls == 3
    assert response.text == "hello"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 1
    assert response.usage_source == "provider"
    assert response.cached is False
    assert seen[0]["seed"] == 4
    assert seen[0]["temperature"] == 0.0
    assert seen[0]["max_tokens"] == 16
    assert seen[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello world"},
    ]
    stored = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "cache").rglob("*.json")
    )
    assert "sk-secret" not in stored

    refused = ScriptedTransport([httpx.Response(400, text="bad key sk-secret")])
    failing = make_client(
        "openai",
        api_key="sk-secret",
        base_url="https://example.invalid/v1",
        transport=refused,
        max_attempts=3,
        backoff_s=0,
    )
    with pytest.raises(LLMClientError) as caught:
        failing.complete(_request(seed=None))
    assert refused.calls == 1
    assert "sk-secret" not in str(caught.value)
    assert "[redacted]" in str(caught.value)


def test_openai_omits_null_seed_and_missing_usage() -> None:
    transport = ScriptedTransport(
        [httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})]
    )
    client = make_client(
        "openai",
        api_key="sk-secret",
        base_url="https://example.invalid/v1",
        transport=transport,
        max_attempts=1,
    )
    response = client.complete(_request(seed=None))
    body = json.loads(transport.requests[0].content.decode("utf-8"))
    assert "seed" not in body
    assert response.prompt_tokens is None
    assert response.usage_source == "missing"
    assert response.cost_usd is None


def test_ollama_maps_eval_counts() -> None:
    transport = ScriptedTransport(
        [
            httpx.Response(
                200,
                json={
                    "model": "llama",
                    "message": {"role": "assistant", "content": "ok"},
                    "prompt_eval_count": 4,
                    "eval_count": 1,
                },
            )
        ]
    )
    client = make_client(
        "ollama",
        base_url="http://127.0.0.1:11434",
        transport=transport,
        max_attempts=1,
    )
    response = client.complete(_request(model="llama", seed=7, temperature=0.0))
    body = json.loads(transport.requests[0].content.decode("utf-8"))
    assert transport.requests[0].url.path == "/api/chat"
    assert body["stream"] is False
    assert body["options"]["seed"] == 7
    assert body["options"]["num_predict"] == 512
    assert response.prompt_tokens == 4
    assert response.completion_tokens == 1
    assert response.usage_source == "provider"
    assert response.text == "ok"


def test_missing_api_key_and_unknown_provider() -> None:
    with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
        make_client("openai", api_key="", base_url="https://example.invalid/v1")
    with pytest.raises(LLMClientError, match="unknown"):
        make_client("somewhere-else")
