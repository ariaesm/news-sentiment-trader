"""Tests for daily aggregation and the price join."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trader.aggregate import (
    assign_trading_day,
    build_panel,
    daily_sentiment,
    load_prices,
    load_scored,
    save_scored,
    scored_frame,
)
from news_sentiment_trader.models import NewsItem, SentimentResult

CAL = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])


def _scored(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "ticker", "score"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_assign_trading_day_cutoff_and_weekend() -> None:
    ts = pd.Series(
        pd.to_datetime(
            [
                "2025-01-02T10:00Z",  # before cutoff -> same day
                "2025-01-02T21:00Z",  # after cutoff -> next trading day
                "2025-01-03T23:00Z",  # Friday evening -> Monday
                "2025-01-04T12:00Z",  # Saturday -> Monday
                "2025-01-08T12:00Z",  # beyond calendar -> NaT
            ],
            utc=True,
        )
    )
    out = assign_trading_day(ts, CAL, cutoff_hour_utc=20)
    assert list(out[:4]) == [CAL[0], CAL[1], CAL[2], CAL[2]]
    assert pd.isna(out.iloc[4])


def test_daily_mean_count_and_decay() -> None:
    scored = _scored(
        [
            ("2025-01-02T10:00Z", "AAA", 0.5),
            ("2025-01-02T11:00Z", "AAA", -0.1),
            ("2025-01-06T10:00Z", "AAA", -1.0),
        ]
    )
    out = daily_sentiment(scored, CAL, halflife_days=1.0).set_index("date")
    assert out.loc["2025-01-02", "sent_mean"] == pytest.approx(0.2)
    assert out.loc["2025-01-02", "news_count"] == 2
    assert np.isnan(out.loc["2025-01-03", "sent_mean"])
    # Decay carries the last value forward through a news-free day.
    assert out.loc["2025-01-03", "sent_decay"] == pytest.approx(0.2)
    # On 01-06 weights are: day-2 items 0.25 each (sum 0.5), today 1.0.
    expected = (0.25 * 0.4 + -1.0) / (0.5 + 1.0)
    assert out.loc["2025-01-06", "sent_decay"] == pytest.approx(expected)


def test_decay_is_causal() -> None:
    """Adding a future headline must not change past aggregated values."""
    base = _scored([("2025-01-02T10:00Z", "AAA", 0.5)])
    extra = _scored([("2025-01-02T10:00Z", "AAA", 0.5), ("2025-01-07T10:00Z", "AAA", -1.0)])
    a = daily_sentiment(base, CAL).set_index("date")
    b = daily_sentiment(extra, CAL).set_index("date")
    pd.testing.assert_frame_equal(a.loc[:"2025-01-06"], b.loc[:"2025-01-06"])


def test_load_prices_long_and_wide(tmp_path: Path) -> None:
    long_path = tmp_path / "long.csv"
    long_path.write_text("date,ticker,close\n2025-01-03,AAA,11\n2025-01-02,AAA,10\n")
    wide_path = tmp_path / "wide.csv"
    wide_path.write_text("date,AAA,BBB\n2025-01-02,10,20\n2025-01-03,11,19\n")
    long_df = load_prices(long_path)
    assert list(long_df["AAA"]) == [10.0, 11.0]
    assert list(load_prices(wide_path).columns) == ["AAA", "BBB"]


def test_build_panel_join_and_returns() -> None:
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.1, 12.1], "ZZZ": [1, 1, 1, 1]}, index=CAL)
    scored = _scored([("2025-01-03T10:00Z", "AAA", 0.4), ("2025-01-03T10:00Z", "NOPX", 0.9)])
    panel = build_panel(scored, prices)
    assert set(panel["ticker"]) == {"AAA"}
    row = panel.set_index("date").loc["2025-01-03"]
    assert row["ret"] == pytest.approx(0.1)
    assert row["sent_mean"] == pytest.approx(0.4)
    assert panel["news_count"].sum() == 1


def test_scored_frame_roundtrip(tmp_path: Path) -> None:
    items = [
        NewsItem("1", datetime(2025, 1, 2, tzinfo=UTC), "s", "t", tickers=("AAA", "BBB")),
        NewsItem("2", datetime(2025, 1, 2, tzinfo=UTC), "s", "untagged"),
    ]
    results = [SentimentResult(0.5, "positive", "r"), SentimentResult(0.0, "neutral")]
    df = scored_frame(items, results)
    assert list(df["ticker"]) == ["AAA", "BBB"]
    path = tmp_path / "out" / "scored.csv"
    save_scored(df, path)
    back = load_scored(path)
    assert back["timestamp"].dt.tz is not None
    assert list(back["score"]) == [0.5, 0.5]
