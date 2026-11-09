"""Turn scored headlines into a daily (date, ticker) sentiment panel joined with prices.

Timing convention (the single most important thing for avoiding look-ahead):

* Prices are daily closes. ``ret`` on date D is the close-to-close return from
  the previous trading day to D.
* A headline is assigned to the first trading day D whose close happens *after*
  the headline was published: news published before ``cutoff_hour_utc`` on a
  trading day belongs to that day; anything later (or on a non-trading day)
  rolls forward to the next trading day.
* Hence sentiment on D is fully known at the close of D, and may only be used
  to predict returns *after* D (``ret`` on D+1 onward).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from news_sentiment_trader.models import NewsItem, SentimentResult

SCORED_COLUMNS = ["id", "timestamp", "source", "title", "ticker", "score", "label", "reason"]


def scored_frame(items: Sequence[NewsItem], results: Sequence[SentimentResult]) -> pd.DataFrame:
    """Combine items and their sentiment into a long frame, one row per (item, ticker).

    Items without tickers are dropped because they cannot be attributed to an asset.
    """
    if len(items) != len(results):
        raise ValueError("items and results must have the same length")
    rows = [
        {
            "id": item.id,
            "timestamp": item.timestamp,
            "source": item.source,
            "title": item.title,
            "ticker": ticker,
            "score": res.score,
            "label": res.label,
            "reason": res.reason,
        }
        for item, res in zip(items, results, strict=True)
        for ticker in item.tickers
    ]
    df = pd.DataFrame(rows, columns=SCORED_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def save_scored(df: pd.DataFrame, path: str | Path) -> None:
    """Write a scored frame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_scored(path: str | Path) -> pd.DataFrame:
    """Read a scored frame written by :func:`save_scored`."""
    df = pd.read_csv(path)
    missing = {"timestamp", "ticker", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"scored file missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load daily closes as a wide frame (index: date, columns: tickers).

    Accepts long format (``date,ticker,close``) or wide format
    (``date,<TICKER1>,<TICKER2>,...``).
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError("price file needs a 'date' column")
    df = df.rename(columns={cols["date"]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "ticker" in cols and "close" in cols:
        df = df.rename(columns={cols["ticker"]: "ticker", cols["close"]: "close"})
        wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    else:
        wide = df.set_index("date")
    wide = wide.sort_index().astype(float)
    wide.columns.name = None
    if wide.index.duplicated().any():
        raise ValueError("duplicate dates in price file")
    return wide


def assign_trading_day(
    timestamps: pd.Series, calendar: pd.DatetimeIndex, cutoff_hour_utc: float = 20.0
) -> pd.Series:
    """Map UTC timestamps to the first trading day at whose close they are known.

    Returns ``NaT`` for timestamps after the last calendar date.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    hours = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    day = ts.dt.tz_localize(None).dt.normalize()
    day = day.where(hours < cutoff_hour_utc, day + pd.Timedelta(days=1))
    cal = pd.DatetimeIndex(calendar).sort_values()
    pos = cal.searchsorted(day.to_numpy(), side="left")
    out = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns]")
    valid = pos < len(cal)
    out[valid] = cal[pos[valid]]
    return out


def daily_sentiment(
    scored: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cutoff_hour_utc: float = 20.0,
    halflife_days: float = 3.0,
) -> pd.DataFrame:
    """Aggregate item scores to a (date, ticker) panel on the trading ``calendar``.

    Columns:
        ``sent_mean``: mean score of items assigned to the day (NaN if none).
        ``news_count``: number of items.
        ``sent_decay``: causal exponentially-decayed average of all past item
        scores with the given half-life (in trading days); carried forward
        through days without news, NaN before the first item.
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    df = scored.copy()
    df["date"] = assign_trading_day(df["timestamp"], cal, cutoff_hour_utc)
    df = df.dropna(subset=["date"])
    decay = 0.5 ** (1.0 / halflife_days)
    frames: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker", sort=True):
        g = grp.groupby("date")["score"].agg(["sum", "count"]).reindex(cal)
        sums = g["sum"].fillna(0.0).to_numpy()
        counts = g["count"].fillna(0).to_numpy()
        w_sum = np.zeros(len(cal))
        ws_sum = np.zeros(len(cal))
        acc_w = acc_ws = 0.0
        for i in range(len(cal)):
            acc_w = acc_w * decay + counts[i]
            acc_ws = acc_ws * decay + sums[i]
            w_sum[i], ws_sum[i] = acc_w, acc_ws
        with np.errstate(invalid="ignore", divide="ignore"):
            decayed = np.where(w_sum > 0, ws_sum / np.where(w_sum > 0, w_sum, 1.0), np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "date": cal,
                    "ticker": ticker,
                    "sent_mean": np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
                    "news_count": counts.astype(int),
                    "sent_decay": decayed,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "sent_mean", "news_count", "sent_decay"])
    return pd.concat(frames, ignore_index=True)


def build_panel(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    cutoff_hour_utc: float = 20.0,
    halflife_days: float = 3.0,
) -> pd.DataFrame:
    """Join daily sentiment with closes and same-day (backward-looking) returns.

    Only tickers present in both the news and the price data are kept.
    """
    tickers = sorted(set(scored["ticker"]).intersection(prices.columns))
    if not tickers:
        raise ValueError("no overlap between news tickers and price columns")
    sent = daily_sentiment(
        scored[scored["ticker"].isin(tickers)], prices.index, cutoff_hour_utc, halflife_days
    )
    px = prices[tickers]
    long_px = px.stack().rename("close").reset_index()
    long_px.columns = ["date", "ticker", "close"]
    rets = px.pct_change(fill_method=None).stack().rename("ret").reset_index()
    rets.columns = ["date", "ticker", "ret"]
    panel = long_px.merge(rets, on=["date", "ticker"], how="left").merge(
        sent, on=["date", "ticker"], how="left"
    )
    panel["news_count"] = panel["news_count"].fillna(0).astype(int)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)
