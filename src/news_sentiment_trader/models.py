"""Core data models shared across the pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any


def to_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to already be in UTC.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def make_id(source: str, title: str, timestamp: datetime) -> str:
    """Build a stable, content-derived identifier for a news item."""
    raw = f"{source}|{title.strip()}|{to_utc(timestamp).isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class NewsItem:
    """A normalized news headline.

    Attributes:
        id: Stable identifier (source-provided or content hash).
        timestamp: Publication time, always timezone-aware UTC.
        source: Publisher or feed name.
        title: Headline text.
        summary: Optional short description / lede.
        tickers: Assets the item refers to (filled by the tagger).
    """

    id: str
    timestamp: datetime
    source: str
    title: str
    summary: str = ""
    tickers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", to_utc(self.timestamp))
        object.__setattr__(self, "tickers", tuple(self.tickers))

    @property
    def text(self) -> str:
        """Title and summary joined, the text that gets scored."""
        return f"{self.title}. {self.summary}".strip(" .") if self.summary else self.title

    def with_tickers(self, tickers: tuple[str, ...] | list[str]) -> NewsItem:
        """Return a copy with ``tickers`` replaced."""
        return replace(self, tickers=tuple(tickers))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "tickers": list(self.tickers),
        }


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Output of a sentiment scorer for a single text.

    Attributes:
        score: Sentiment in [-1, 1]; negative is bearish, positive bullish.
        label: ``"positive"``, ``"negative"`` or ``"neutral"``.
        reason: Short human-readable justification.
    """

    score: float
    label: str
    reason: str = ""

    def __post_init__(self) -> None:
        clipped = max(-1.0, min(1.0, float(self.score)))
        object.__setattr__(self, "score", clipped)


def label_for(score: float, band: float = 0.05) -> str:
    """Map a numeric score to a discrete label using a neutral ``band``."""
    if score > band:
        return "positive"
    if score < -band:
        return "negative"
    return "neutral"
