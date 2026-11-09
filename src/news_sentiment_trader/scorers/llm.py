"""LLM sentiment scorer for any OpenAI-compatible chat completions API.

Works with OpenAI, OpenRouter, and local servers such as Ollama
(``http://localhost:11434/v1``). Uses only the standard library for HTTP.

Design notes:

* Headlines are sent in batches; the model must answer with strict JSON
  ``{"results": [{"id": int, "score": float, "label": str, "reason": str}, ...]}``.
* Output is parsed defensively (code fences, surrounding prose, out-of-range
  scores, label/score disagreement). A batch that cannot be parsed is retried,
  then each headline is retried on its own.
* Results are cached on disk keyed by ``sha256(model + text)`` so re-running a
  pipeline over the same news costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from news_sentiment_trader.models import SentimentResult, label_for

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]

VALID_LABELS = ("positive", "negative", "neutral")

SYSTEM_PROMPT = """You are a financial news sentiment analyst.
For each numbered headline, judge the likely short-term price impact on the
asset(s) it mentions, from the perspective of an investor holding that asset.

Respond with ONLY a JSON object, no prose, in exactly this schema:
{"results": [{"id": <int>, "score": <float in [-1, 1]>,
              "label": "positive" | "negative" | "neutral",
              "reason": "<= 20 words"}]}

Rules:
- One result per headline, same ids as given.
- score -1 = very bearish, 0 = neutral / no impact, 1 = very bullish.
- label must agree with the sign of score (|score| <= 0.05 is neutral).
- Judge only the text; do not use knowledge of what happened afterwards."""


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot be reached or returns garbage."""


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(slots=True)
class LLMConfig:
    """Connection and behaviour settings for :class:`LLMScorer`."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    batch_size: int = 10
    max_retries: int = 3
    backoff_seconds: float = 1.0
    timeout: float = 60.0
    temperature: float = 0.0
    json_mode: bool = True
    cache_dir: Path | None = field(default_factory=lambda: Path(".cache/llm"))

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMConfig:
        """Read ``NST_LLM_BASE_URL``, ``NST_LLM_API_KEY`` and ``NST_LLM_MODEL``."""
        cfg = cls(
            base_url=os.environ.get("NST_LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("NST_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            model=os.environ.get("NST_LLM_MODEL", DEFAULT_MODEL),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


def urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cache_key(model: str, text: str) -> str:
    """Stable cache key for a (model, text) pair."""
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class DiskCache:
    """Tiny JSON-file-per-entry cache (safe to commit, easy to inspect)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> SentimentResult | None:
        """Return the cached result for ``key`` or ``None``."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return SentimentResult(score=float(d["score"]), label=d["label"], reason=d["reason"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set(self, key: str, result: SentimentResult) -> None:
        """Store ``result`` under ``key``."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"score": result.score, "label": result.label, "reason": result.reason}
        path.write_text(json.dumps(payload), encoding="utf-8")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(content: str) -> Any:
    """Extract the first JSON object/array from an LLM message.

    Handles Markdown code fences and leading/trailing prose.
    """
    text = content.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[min(starts) :])
    return obj


def coerce_result(obj: Any) -> SentimentResult:
    """Validate one ``{"score", "label", "reason"}`` object.

    The numeric score is authoritative: it is clipped to [-1, 1] and, if the
    model's label disagrees with its sign, the label is recomputed.
    """
    if not isinstance(obj, dict) or "score" not in obj:
        raise ValueError(f"invalid result object: {obj!r}")
    score = float(obj["score"])
    if score != score:  # NaN
        raise ValueError("score is NaN")
    score = max(-1.0, min(1.0, score))
    label = str(obj.get("label", "")).strip().lower()
    expected = label_for(score)
    if label not in VALID_LABELS or label != expected:
        label = expected
    reason = str(obj.get("reason", ""))[:300]
    return SentimentResult(score=score, label=label, reason=reason)


def parse_batch(content: str, n: int) -> list[SentimentResult]:
    """Parse a batched response for ``n`` headlines (ids ``0..n-1``)."""
    data = extract_json(content)
    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        elif n == 1 and "score" in data:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("expected a list of results")
    by_id: dict[int, SentimentResult] = {}
    for pos, obj in enumerate(data):
        idx = int(obj.get("id", pos)) if isinstance(obj, dict) else pos
        by_id[idx] = coerce_result(obj)
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise ValueError(f"missing results for ids {missing}")
    return [by_id[i] for i in range(n)]


class LLMScorer:
    """Score headlines with an LLM via an OpenAI-compatible ``/chat/completions`` API.

    Args:
        config: Endpoint, model and behaviour settings.
        transport: Callable performing the HTTP POST; injectable for tests.
        sleep: Sleep function used for retry backoff; injectable for tests.
    """

    name = "llm"

    def __init__(
        self,
        config: LLMConfig | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.transport: Transport = transport or urllib_transport
        self.sleep = sleep
        self.cache = DiskCache(self.config.cache_dir) if self.config.cache_dir else None
        self.api_calls = 0

    # -- HTTP -------------------------------------------------------------
    def _request(self, texts: Sequence[str]) -> str:
        cfg = self.config
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        payload: dict[str, Any] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Headlines:\n{numbered}"},
            ],
        }
        if cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.api_calls += 1
        response = self.transport(url, headers, payload, cfg.timeout)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected response shape: {response!r}"[:300]) from exc

    def _score_batch_with_retries(self, texts: Sequence[str]) -> list[SentimentResult]:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return parse_batch(self._request(texts), len(texts))
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise LLMError(f"HTTP {exc.code} from LLM endpoint") from exc
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_err = exc
            if attempt < self.config.max_retries - 1:
                self.sleep(self.config.backoff_seconds * (2**attempt))
        raise LLMError(f"LLM scoring failed after retries: {last_err}") from last_err

    # -- public API ---------------------------------------------------------
    def score_texts(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Score texts, using the cache and batching only the misses."""
        results: list[SentimentResult | None] = [None] * len(texts)
        todo: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(cache_key(self.config.model, text)) if self.cache else None
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)

        size = max(1, self.config.batch_size)
        for start in range(0, len(todo), size):
            idxs = todo[start : start + size]
            batch = [texts[i] for i in idxs]
            try:
                scored = self._score_batch_with_retries(batch)
            except LLMError:
                if len(batch) == 1:
                    raise
                # Fall back to one-by-one so one bad headline does not sink the batch.
                scored = [self._score_batch_with_retries([t])[0] for t in batch]
            for i, res in zip(idxs, scored, strict=True):
                results[i] = res
                if self.cache:
                    self.cache.set(cache_key(self.config.model, texts[i]), res)
        return [r for r in results if r is not None]
