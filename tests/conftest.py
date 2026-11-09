"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from news_sentiment_trader.models import NewsItem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    """Path to the repository root."""
    return ROOT


@pytest.fixture
def sample_items() -> list[NewsItem]:
    """A handful of hand-written news items."""
    return [
        NewsItem(
            id="a",
            timestamp=datetime(2025, 1, 2, 14, 0, tzinfo=UTC),
            source="wire",
            title="Apple beats earnings estimates on strong iPhone demand",
        ),
        NewsItem(
            id="b",
            timestamp=datetime(2025, 1, 2, 22, 30, tzinfo=UTC),
            source="wire",
            title="Bitcoin plunges as exchange hack sparks selloff",
        ),
        NewsItem(
            id="c",
            timestamp=datetime(2025, 1, 3, 9, 15, tzinfo=UTC),
            source="blog",
            title="Microsoft and Apple announce cloud partnership",
        ),
    ]
