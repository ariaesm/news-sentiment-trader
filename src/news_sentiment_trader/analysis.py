"""Statistical analysis: does daily sentiment lead returns?

All functions take the long ``(date, ticker)`` panel produced by
:func:`news_sentiment_trader.aggregate.build_panel`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

POOLED = "ALL"


def add_forward_returns(panel: pd.DataFrame, horizons: Iterable[int] = (1, 5)) -> pd.DataFrame:
    """Add ``fwd_ret_{k}`` = close[t+k] / close[t] - 1 for each ticker.

    These are *targets*: they look into the future by construction and must
    never be used as features.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker")["close"]
    for k in horizons:
        if k < 1:
            raise ValueError("horizons must be >= 1")
        out[f"fwd_ret_{k}"] = grouped.shift(-k) / out["close"] - 1.0
    return out


def _corr_row(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        nan = float("nan")
        return {"n": n, "pearson": nan, "pearson_p": nan, "spearman": nan, "spearman_p": nan}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {
        "n": n,
        "pearson": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def correlations(
    panel: pd.DataFrame, signal: str = "sent_mean", horizons: Sequence[int] = (1, 5)
) -> pd.DataFrame:
    """Pearson and Spearman correlation of ``signal`` with k-day forward returns.

    Returns one row per (ticker, horizon), plus pooled rows with ticker ``"ALL"``.
    Overlapping k>1 forward returns are autocorrelated, so their p-values are
    optimistic; treat them as descriptive.
    """
    df = add_forward_returns(panel, horizons)
    rows: list[dict[str, object]] = []
    groups = [(t, g) for t, g in df.groupby("ticker")] + [(POOLED, df)]
    for ticker, g in groups:
        for k in horizons:
            rows.append({"ticker": ticker, "horizon": k, **_corr_row(g[signal], g[f"fwd_ret_{k}"])})
    return pd.DataFrame(rows)


def lead_lag(
    panel: pd.DataFrame, signal: str = "sent_mean", lags: Iterable[int] = range(-5, 6)
) -> pd.DataFrame:
    """Correlation of ``signal`` on day t with the 1-day return on day t+lag.

    ``lag > 0``: sentiment *leads* returns (the interesting, tradable case).
    ``lag = 0``: same-day return (news and price react together).
    ``lag < 0``: returns lead sentiment (news chasing price).

    Returns a frame with columns ``lag, ticker, pearson, spearman, n``.
    """
    rows: list[dict[str, object]] = []
    df = panel.sort_values(["ticker", "date"])
    lag_list = list(lags)
    shifted: dict[int, pd.Series] = {
        lag: df.groupby("ticker")["ret"].shift(-lag) for lag in lag_list
    }
    for lag in lag_list:
        tmp = df.assign(target=shifted[lag])
        groups = [(t, g) for t, g in tmp.groupby("ticker")] + [(POOLED, tmp)]
        for ticker, g in groups:
            r = _corr_row(g[signal], g["target"])
            rows.append(
                {
                    "lag": lag,
                    "ticker": ticker,
                    "pearson": r["pearson"],
                    "spearman": r["spearman"],
                    "n": r["n"],
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    """Average cumulative return paths after extreme-sentiment days."""

    table: pd.DataFrame
    pos_threshold: float
    neg_threshold: float
    n_pos: int
    n_neg: int


def _mean_ci(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def event_study(
    panel: pd.DataFrame,
    signal: str = "sent_mean",
    quantile: float = 0.9,
    window: int = 5,
    confidence: float = 0.95,
) -> EventStudyResult:
    """Cumulative returns over days 1..window after strongly positive/negative days.

    Events are days where ``signal`` is above its ``quantile`` (positive) or below
    its ``1 - quantile`` quantile (negative), computed over all news days. Using
    full-sample quantiles is fine for *description* but is itself in-sample;
    the backtest uses fixed thresholds instead.
    """
    df = add_forward_returns(panel, range(1, window + 1))
    news = df[df[signal].notna()]
    if news.empty:
        raise ValueError("no news days in panel")
    hi = float(news[signal].quantile(quantile))
    lo = float(news[signal].quantile(1 - quantile))
    pos = news[news[signal] >= hi]
    neg = news[news[signal] <= lo]
    rows = [
        {"day": 0, **dict.fromkeys(("pos", "pos_lo", "pos_hi", "neg", "neg_lo", "neg_hi"), 0.0)}
    ]
    for k in range(1, window + 1):
        col = f"fwd_ret_{k}"
        pm, plo, phi = _mean_ci(pos[col].to_numpy(dtype=float), confidence)
        nm, nlo, nhi = _mean_ci(neg[col].to_numpy(dtype=float), confidence)
        rows.append(
            {
                "day": k,
                "pos": pm,
                "pos_lo": plo,
                "pos_hi": phi,
                "neg": nm,
                "neg_lo": nlo,
                "neg_hi": nhi,
            }
        )
    table = pd.DataFrame(rows)
    table["spread"] = table["pos"] - table["neg"]
    return EventStudyResult(table, hi, lo, len(pos), len(neg))


def hit_rate(
    panel: pd.DataFrame, signal: str = "sent_mean", horizon: int = 1, min_abs: float = 0.0
) -> dict[str, float]:
    """Share of news days where sign(signal) matches sign(forward return).

    Days with ``|signal| <= min_abs`` or a zero forward return are ignored.
    Returns the hit rate, the number of days and a one-sided binomial p-value
    against the 50% coin-flip null.
    """
    df = add_forward_returns(panel, [horizon])
    col = f"fwd_ret_{horizon}"
    d = df[df[signal].notna() & df[col].notna()]
    d = d[(d[signal].abs() > min_abs) & (d[col] != 0)]
    n = len(d)
    if n == 0:
        return {"hit_rate": float("nan"), "n": 0, "p_value": float("nan")}
    hits = int((np.sign(d[signal]) == np.sign(d[col])).sum())
    p = float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue)
    return {"hit_rate": hits / n, "n": n, "p_value": p}


def summarize(
    panel: pd.DataFrame,
    signal: str = "sent_mean",
    horizons: Sequence[int] = (1, 5),
    lags: Iterable[int] = range(-5, 6),
) -> dict[str, object]:
    """Run every analysis and return the results in one dict."""
    return {
        "correlations": correlations(panel, signal, horizons),
        "lead_lag": lead_lag(panel, signal, lags),
        "event_study": event_study(panel, signal),
        "hit_rate": hit_rate(panel, signal),
    }
