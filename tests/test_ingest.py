"""Tests for news ingestion."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from news_sentiment_trader.ingest import (
    deduplicate,
    load_csv,
    load_json,
    load_news,
    parse_rss,
    parse_timestamp,
    save_json,
)
from news_sentiment_trader.models import NewsItem

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test Wire</title>
<item><title>Stocks rally on rate cut hopes</title><guid>g1</guid>
<description>Markets rose.</description><pubDate>Tue, 07 Jan 2025 15:30:00 -0500</pubDate></item>
<item><title>No date here</title></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom Wire</title>
<entry><id>urn:1</id><title>Oil slumps</title><updated>2025-01-08T10:00:00Z</updated>
<summary>Crude fell.</summary></entry></feed>"""


def test_parse_timestamp_formats() -> None:
    expected = datetime(2025, 1, 7, 20, 30, tzinfo=UTC)
    assert parse_timestamp("2025-01-07T20:30:00Z") == expected
    assert parse_timestamp("2025-01-07T15:30:00-05:00") == expected
    assert parse_timestamp("Tue, 07 Jan 2025 15:30:00 -0500") == expected
    assert parse_timestamp(expected.timestamp()) == expected
    with pytest.raises(ValueError):
        parse_timestamp("yesterday-ish")


def test_newsitem_normalizes_to_utc() -> None:
    tz = timezone(timedelta(hours=3, minutes=30))
    item = NewsItem(id="x", timestamp=datetime(2025, 1, 1, 3, 30, tzinfo=tz), source="s", title="t")
    assert item.timestamp == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    naive = NewsItem(id="y", timestamp=datetime(2025, 1, 1, 12), source="s", title="t")
    assert naive.timestamp.tzinfo is UTC


def test_json_roundtrip(tmp_path: Path, sample_items: list[NewsItem]) -> None:
    path = tmp_path / "news.json"
    save_json(sample_items, path)
    loaded = load_json(path)
    assert loaded == sample_items


def test_json_items_wrapper_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "n.json"
    payload = {"items": [{"headline": "Tesla recalls cars", "published": "2025-02-01T00:00:00Z"}]}
    path.write_text(json.dumps(payload))
    [item] = load_json(path)
    assert item.title == "Tesla recalls cars"
    assert item.source == "unknown"
    assert len(item.id) == 16


def test_load_csv_with_tickers(tmp_path: Path) -> None:
    path = tmp_path / "n.csv"
    path.write_text(
        "timestamp,source,title,tickers\n2025-01-02T10:00:00Z,wire,Apple up,aapl;msft\n"
    )
    [item] = load_csv(path)
    assert item.tickers == ("AAPL", "MSFT")


def test_parse_rss_skips_undated_items() -> None:
    items = parse_rss(RSS)
    assert len(items) == 1
    assert items[0].id == "g1"
    assert items[0].source == "Test Wire"
    assert items[0].timestamp == datetime(2025, 1, 7, 20, 30, tzinfo=UTC)


def test_parse_atom() -> None:
    [item] = parse_rss(ATOM)
    assert item.title == "Oil slumps"
    assert item.summary == "Crude fell."


def test_load_news_dispatch(tmp_path: Path) -> None:
    rss = tmp_path / "feed.xml"
    rss.write_text(RSS)
    assert len(load_news(rss)) == 1
    with pytest.raises(ValueError):
        load_news(tmp_path / "news.parquet")


def test_deduplicate_keeps_earliest(sample_items: list[NewsItem]) -> None:
    later = NewsItem(
        id="dup",
        timestamp=sample_items[0].timestamp + timedelta(hours=1),
        source="WIRE",
        title=sample_items[0].title.upper(),
    )
    out = deduplicate([later, *sample_items])
    assert [i.id for i in out] == ["a", "b", "c"]
