"""Tests for the LLM scorer. All HTTP is mocked; no network is used."""

from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from news_sentiment_trader.scorers import SentimentScorer
from news_sentiment_trader.scorers.llm import (
    LLMConfig,
    LLMError,
    LLMScorer,
    coerce_result,
    extract_json,
    parse_batch,
)


def _reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeTransport:
    """Scores headlines containing 'up' as +0.8 and others as -0.4."""

    def __init__(self, failures: list[Exception | str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failures = list(failures or [])

    def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return _reply(failure)
        user = payload["messages"][1]["content"]
        lines = re.findall(r"^(\d+)\. (.*)$", user, flags=re.MULTILINE)
        results = [
            {
                "id": int(i),
                "score": 0.8 if "up" in text else -0.4,
                "label": "positive" if "up" in text else "negative",
                "reason": "mock",
            }
            for i, text in lines
        ]
        return _reply(json.dumps({"results": results}))


def _scorer(tmp_path: Path, transport: FakeTransport, **kw: Any) -> LLMScorer:
    cfg = LLMConfig(
        base_url="http://localhost:11434/v1/",
        api_key="k",
        model="test-model",
        cache_dir=tmp_path / "cache",
        **kw,
    )
    return LLMScorer(cfg, transport=transport, sleep=lambda _s: None)


def test_protocol_and_request_shape(tmp_path: Path) -> None:
    transport = FakeTransport()
    scorer = _scorer(tmp_path, transport)
    assert isinstance(scorer, SentimentScorer)
    out = scorer.score_texts(["stocks up", "stocks down"])
    assert [r.score for r in out] == [0.8, -0.4]
    call = transport.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["payload"]["model"] == "test-model"
    assert call["payload"]["response_format"] == {"type": "json_object"}


def test_batching(tmp_path: Path) -> None:
    transport = FakeTransport()
    scorer = _scorer(tmp_path, transport, batch_size=2)
    out = scorer.score_texts([f"item {i} up" for i in range(5)])
    assert len(out) == 5
    assert len(transport.calls) == 3


def test_cache_makes_reruns_free(tmp_path: Path) -> None:
    transport = FakeTransport()
    texts = ["btc up", "aapl down"]
    first = _scorer(tmp_path, transport).score_texts(texts)
    assert len(transport.calls) == 1
    second_scorer = _scorer(tmp_path, transport)
    second = second_scorer.score_texts([*texts, "msft up"])
    assert second[:2] == first
    assert second_scorer.api_calls == 1  # only the new headline was requested
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 3


def test_cache_is_keyed_by_model(tmp_path: Path) -> None:
    transport = FakeTransport()
    _scorer(tmp_path, transport).score_texts(["btc up"])
    other = LLMScorer(
        LLMConfig(model="other-model", cache_dir=tmp_path / "cache"),
        transport=transport,
        sleep=lambda _s: None,
    )
    other.score_texts(["btc up"])
    assert len(transport.calls) == 2


def test_retries_on_garbage_and_transient_http(tmp_path: Path) -> None:
    http_503 = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b""))  # type: ignore[arg-type]
    transport = FakeTransport(failures=["I think it's bullish!", http_503])
    sleeps: list[float] = []
    cfg = LLMConfig(cache_dir=None, max_retries=3, backoff_seconds=0.5)
    scorer = LLMScorer(cfg, transport=transport, sleep=sleeps.append)
    [res] = scorer.score_texts(["gold up"])
    assert res.score == 0.8
    assert sleeps == [0.5, 1.0]


def test_non_retryable_http_error(tmp_path: Path) -> None:
    http_401 = urllib.error.HTTPError("u", 401, "nope", {}, io.BytesIO(b""))  # type: ignore[arg-type]
    scorer = _scorer(tmp_path, FakeTransport(failures=[http_401]))
    with pytest.raises(LLMError, match="401"):
        scorer.score_texts(["x up"])


def test_batch_failure_falls_back_to_single(tmp_path: Path) -> None:
    bad = json.dumps({"results": [{"id": 0, "score": 0.1}]})  # missing id 1
    transport = FakeTransport(failures=[bad, bad])
    scorer = _scorer(tmp_path, transport, max_retries=2)
    out = scorer.score_texts(["a up", "b down"])
    assert [r.score for r in out] == [0.8, -0.4]
    assert len(transport.calls) == 4


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n{"score": 0.5}\n```') == {"score": 0.5}
    assert extract_json('Sure! Here you go: [{"score": 1}] hope it helps') == [{"score": 1}]
    with pytest.raises(ValueError):
        extract_json("no json at all")


def test_coerce_result_clips_and_fixes_label() -> None:
    res = coerce_result({"score": 3, "label": "negative", "reason": "r"})
    assert res.score == 1.0 and res.label == "positive"
    assert coerce_result({"score": "0.0", "label": "bullish"}).label == "neutral"
    with pytest.raises(ValueError):
        coerce_result({"label": "positive"})


def test_parse_batch_single_object_and_ordering() -> None:
    assert parse_batch('{"score": -0.5, "label": "negative", "reason": ""}', 1)[0].score == -0.5
    content = json.dumps([{"id": 1, "score": 0.2}, {"id": 0, "score": -0.2}])
    assert [r.score for r in parse_batch(content, 2)] == [-0.2, 0.2]


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("NST_LLM_MODEL", "llama3.1")
    monkeypatch.setenv("NST_LLM_API_KEY", "ollama")
    cfg = LLMConfig.from_env(batch_size=5, model=None)
    assert (cfg.base_url, cfg.model, cfg.api_key, cfg.batch_size) == (
        "http://localhost:11434/v1",
        "llama3.1",
        "ollama",
        5,
    )
