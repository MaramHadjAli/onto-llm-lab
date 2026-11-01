"""One LLM client: disk cache, recorded sampling parameters, token log, retries.

Providers are OpenAI-compatible chat completions, Ollama ``/api/chat``, and
``MockProvider``. The mock is deterministic and does not open a socket.
Token counts from the mock are a local whitespace estimate, marked
``usage_source="whitespace_estimate"``. Provider responses that omit usage
keep ``prompt_tokens`` as ``None`` rather than inventing a count.

Cost is computed only when the caller supplies a ``TokenPrices`` table.
No model price list is built into this module. Cached rows keep the cost
stored when the completion was first produced. Sum only rows with
``cached`` false when estimating new spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

UsageSource = Literal["provider", "missing", "whitespace_estimate"]
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_PROTECTED_FIELDS = {"model", "messages", "temperature", "max_tokens", "seed"}


class LLMClientError(RuntimeError):
    """The client could not produce a completion."""


class TokenPrices(BaseModel):
    """Caller-supplied USD prices per million tokens."""

    model_config = ConfigDict(frozen=True)

    prompt_usd_per_million: float
    completion_usd_per_million: float

    @field_validator("prompt_usd_per_million", "completion_usd_per_million")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("token prices must be non-negative")
        return value


class LLMRequest(BaseModel):
    """One sampled completion request. ``extra`` is merged into the HTTP body."""

    model_config = ConfigDict(frozen=True)

    model: str
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int = 512
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_tokens must be positive")
        return value

    @field_validator("extra")
    @classmethod
    def _extra_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        overlap = _PROTECTED_FIELDS & set(value)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"extra must not override {names}")
        json.dumps(value, sort_keys=True)
        return dict(value)


class LLMResponse(BaseModel):
    """A completion plus the parameters and usage needed to reproduce it."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    provider: str
    endpoint: str
    prompt_hash: str
    cache_key: str
    temperature: float
    seed: int | None
    max_tokens: int
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: UsageSource
    cost_usd: float | None
    cached: bool


class RawCompletion(BaseModel):
    """Provider payload before cache and cost are attached."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: UsageSource


class CompletionProvider(Protocol):
    """Uncached completion source."""

    name: str
    endpoint: str

    def complete(self, request: LLMRequest) -> RawCompletion:
        """Return one completion or raise ``LLMClientError``."""


def hash_prompt(request: LLMRequest) -> str:
    """SHA-256 of the system text and the user prompt."""
    raw = (request.system or "") + "\n---\n" + request.prompt
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_key(provider: str, endpoint: str, request: LLMRequest) -> str:
    """SHA-256 of provider, endpoint, model, sampling parameters, and prompt hash."""
    payload = {
        "provider": provider,
        "endpoint": endpoint,
        "model": request.model,
        "prompt_hash": hash_prompt(request),
        "temperature": request.temperature,
        "seed": request.seed,
        "max_tokens": request.max_tokens,
        "extra": request.extra,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prices: TokenPrices | None,
) -> float | None:
    """Return USD cost, or ``None`` when usage or prices are absent."""
    if prices is None or prompt_tokens is None or completion_tokens is None:
        return None
    total = (
        prompt_tokens * prices.prompt_usd_per_million
        + completion_tokens * prices.completion_usd_per_million
    )
    return total / 1_000_000


class DiskCache:
    """JSON cache. One file per key. Writes are replaced atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise LLMClientError(f"cache entry is not an object: {path}")
        return loaded

    def put(self, key: str, payload: Mapping[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"


class MockProvider:
    """Deterministic offline provider. The same request returns the same text."""

    name = "mock"
    endpoint = "mock://local"

    def __init__(self, responder: Callable[[LLMRequest], str] | None = None) -> None:
        self._responder = responder or default_mock_text

    def complete(self, request: LLMRequest) -> RawCompletion:
        text = self._responder(request)
        return RawCompletion(
            text=text,
            model=request.model,
            prompt_tokens=_whitespace_tokens(request.system, request.prompt),
            completion_tokens=_whitespace_tokens(text),
            usage_source="whitespace_estimate",
        )


def default_mock_text(request: LLMRequest) -> str:
    """Stable text that depends on the prompt, seed, and temperature."""
    digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:12]
    seed = "none" if request.seed is None else str(request.seed)
    return f"mock-response seed={seed} temperature={request.temperature} digest={digest}"


class OpenAICompatibleProvider:
    """Chat completions at ``{base_url}/chat/completions``."""

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 60.0,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY is not set")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.endpoint = base_url.rstrip("/")
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._client = httpx.Client(
            base_url=self.endpoint,
            transport=transport,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def complete(self, request: LLMRequest) -> RawCompletion:
        payload: dict[str, Any] = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": _messages(request),
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        payload.update(request.extra)
        response = _post_json(
            self._client,
            "/chat/completions",
            payload,
            attempts=self._max_attempts,
            backoff_s=self._backoff_s,
            secret=self._api_key,
        )
        return _parse_openai(response.json(), fallback_model=request.model)

    def close(self) -> None:
        self._client.close()


class OllamaProvider:
    """Local Ollama chat. ``prompt_eval_count`` and ``eval_count`` map to tokens."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 120.0,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.endpoint = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._client = httpx.Client(base_url=self.endpoint, transport=transport, timeout=timeout_s)

    def complete(self, request: LLMRequest) -> RawCompletion:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        options.update(request.extra)
        payload = {
            "model": request.model,
            "messages": _messages(request),
            "stream": False,
            "options": options,
        }
        response = _post_json(
            self._client,
            "/api/chat",
            payload,
            attempts=self._max_attempts,
            backoff_s=self._backoff_s,
            secret="",
        )
        return _parse_ollama(response.json(), fallback_model=request.model)

    def close(self) -> None:
        self._client.close()


class LLMClient:
    """Cache and usage log around a single provider."""

    def __init__(
        self,
        provider: CompletionProvider,
        *,
        cache_dir: Path | None = None,
        usage_log: Path | None = None,
        prices: TokenPrices | None = None,
    ) -> None:
        self.provider = provider
        self.cache = DiskCache(cache_dir) if cache_dir is not None else None
        self.usage_log = usage_log
        self.prices = prices

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a cached completion when the key matches, otherwise call the provider."""
        key = cache_key(self.provider.name, self.provider.endpoint, request)
        prompt_hash = hash_prompt(request)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                stored = LLMResponse.model_validate(cached["response"])
                replayed = stored.model_copy(update={"cached": True})
                self._log(replayed)
                return replayed
        raw = self.provider.complete(request)
        response = LLMResponse(
            text=raw.text,
            model=raw.model,
            provider=self.provider.name,
            endpoint=self.provider.endpoint,
            prompt_hash=prompt_hash,
            cache_key=key,
            temperature=request.temperature,
            seed=request.seed,
            max_tokens=request.max_tokens,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            usage_source=raw.usage_source,
            cost_usd=estimate_cost(raw.prompt_tokens, raw.completion_tokens, self.prices),
            cached=False,
        )
        if self.cache is not None:
            self.cache.put(
                key,
                {
                    "request": request.model_dump(mode="json"),
                    "response": response.model_dump(mode="json"),
                },
            )
        self._log(response)
        return response

    def _log(self, response: LLMResponse) -> None:
        if self.usage_log is None:
            return
        record = {
            "cached": response.cached,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endpoint": response.endpoint,
            "max_tokens": response.max_tokens,
            "model": response.model,
            "prompt_hash": response.prompt_hash,
            "prompt_tokens": response.prompt_tokens,
            "provider": response.provider,
            "seed": response.seed,
            "temperature": response.temperature,
            "usage_source": response.usage_source,
        }
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def make_client(
    provider: str = "mock",
    *,
    cache_dir: Path | None = None,
    usage_log: Path | None = None,
    prices: TokenPrices | None = None,
    transport: httpx.BaseTransport | None = None,
    max_attempts: int = 3,
    backoff_s: float = 0.5,
    api_key: str | None = None,
    base_url: str | None = None,
    responder: Callable[[LLMRequest], str] | None = None,
) -> LLMClient:
    """Build the shared client.

    ``provider`` is ``mock``, ``openai``, or ``ollama``. The default is mock
    so a fresh process does not call the network. The model name belongs on
    each ``LLMRequest``.
    """
    if provider == "mock":
        backend: CompletionProvider = MockProvider(responder)
    elif provider == "openai":
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        backend = OpenAICompatibleProvider(
            base_url=resolved_url,
            api_key=resolved_key,
            transport=transport,
            max_attempts=max_attempts,
            backoff_s=backoff_s,
        )
    elif provider == "ollama":
        resolved_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        backend = OllamaProvider(
            base_url=resolved_url,
            transport=transport,
            max_attempts=max_attempts,
            backoff_s=backoff_s,
        )
    else:
        raise LLMClientError(f"unknown LLM provider: {provider}")
    return LLMClient(backend, cache_dir=cache_dir, usage_log=usage_log, prices=prices)


def _messages(request: LLMRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    return messages


def _post_json(
    client: httpx.Client,
    url: str,
    payload: Mapping[str, Any],
    *,
    attempts: int,
    backoff_s: float,
    secret: str,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post(url, json=dict(payload))
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if response.status_code not in _RETRYABLE_STATUS:
                if response.status_code >= 400:
                    body = _redact(response.text[:300], secret)
                    raise LLMClientError(f"HTTP {response.status_code}: {body}") from None
                return response
            last_error = LLMClientError(f"HTTP {response.status_code}")
        if attempt + 1 < attempts and backoff_s > 0:
            time.sleep(backoff_s * (2**attempt))
    raise LLMClientError(f"request failed after {attempts} attempts") from last_error


def _parse_openai(data: Any, *, fallback_model: str) -> RawCompletion:
    if not isinstance(data, dict):
        raise LLMClientError("OpenAI response is not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError("OpenAI response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMClientError("OpenAI message content is not a string")
    usage = data.get("usage")
    prompt_tokens, completion_tokens, source = _usage_pair(
        usage if isinstance(usage, dict) else {},
        "prompt_tokens",
        "completion_tokens",
    )
    model = data.get("model")
    return RawCompletion(
        text=content,
        model=model if isinstance(model, str) and model else fallback_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_source=source,
    )


def _parse_ollama(data: Any, *, fallback_model: str) -> RawCompletion:
    if not isinstance(data, dict):
        raise LLMClientError("Ollama response is not a JSON object")
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMClientError("Ollama message content is not a string")
    prompt_tokens, completion_tokens, source = _usage_pair(data, "prompt_eval_count", "eval_count")
    model = data.get("model")
    return RawCompletion(
        text=content,
        model=model if isinstance(model, str) and model else fallback_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_source=source,
    )


def _usage_pair(
    payload: Mapping[str, Any],
    prompt_key: str,
    completion_key: str,
) -> tuple[int | None, int | None, UsageSource]:
    prompt = payload.get(prompt_key)
    completion = payload.get(completion_key)
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion, "provider"
    return None, None, "missing"


def _whitespace_tokens(*parts: str | None) -> int:
    text = " ".join(part for part in parts if part)
    if not text.strip():
        return 0
    return len(text.split())


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text
