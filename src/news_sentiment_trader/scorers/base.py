"""Scorer protocol and helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from news_sentiment_trader.models import NewsItem, SentimentResult


@runtime_checkable
class SentimentScorer(Protocol):
    """Anything that maps texts to :class:`SentimentResult` objects."""

    name: str

    def score_texts(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Score each text independently; output order matches input order."""
        ...


def score_items(scorer: SentimentScorer, items: Sequence[NewsItem]) -> list[SentimentResult]:
    """Score the text (title + summary) of each news item."""
    return scorer.score_texts([i.text for i in items])
