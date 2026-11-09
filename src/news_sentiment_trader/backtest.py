"""Vectorised daily backtest of a threshold sentiment signal.

Timing: the position held over day t's return (close[t-1] -> close[t]) is
decided from the signal observed on day t-1, which by construction of the
panel only contains news known at close[t-1]. That one-day shift is the whole
look-ahead defence, and :func:`positions` is the only place it happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Strategy parameters.

    Attributes:
        signal: Panel column used as the signal.
        threshold: Go long above ``threshold``; short (or flat) below ``-threshold``.
        allow_short: If False, negative signals mean flat instead of short.
        cost_bps: One-way transaction cost in basis points per unit of turnover.
        periods_per_year: Used to annualise Sharpe (252 equities, 365 crypto).
    """

    signal: str = "sent_mean"
    threshold: float = 0.2
    allow_short: bool = True
    cost_bps: float = 5.0
    periods_per_year: int = 252


@dataclass(slots=True)
class BacktestResult:
    """Daily returns, equity curves and summary metrics."""

    daily: pd.DataFrame
    metrics: pd.DataFrame
    per_ticker: dict[str, pd.DataFrame] = field(default_factory=dict)


def positions(signal: pd.Series, threshold: float, allow_short: bool = True) -> pd.Series:
    """Map a signal series (indexed by date) to positions applied to the *next* day.

    Missing signal (no news) means flat.
    """
    s = signal.fillna(0.0)
    raw = pd.Series(0.0, index=s.index)
    raw[s > threshold] = 1.0
    if allow_short:
        raw[s < -threshold] = -1.0
    return raw.shift(1).fillna(0.0)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative fraction."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def performance(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Total return, annualised return/vol, Sharpe (rf=0), max drawdown, hit rate."""
    r = returns.dropna()
    if r.empty:
        nan = float("nan")
        return dict.fromkeys(
            ["total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"], nan
        )
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = len(r) / periods_per_year
    ann_ret = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if vol else float("nan")
    active = r[r != 0]
    hit = float((active > 0).mean()) if len(active) else float("nan")
    return {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit,
    }


def run_ticker(panel_t: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Backtest one ticker. ``panel_t`` needs ``date``, ``ret`` and the signal column."""
    df = panel_t.sort_values("date").set_index("date")
    pos = positions(df[cfg.signal], cfg.threshold, cfg.allow_short)
    ret = df["ret"].fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cfg.cost_bps / 10_000
    strat = pos * ret - cost
    return pd.DataFrame(
        {"signal": df[cfg.signal], "position": pos, "ret": ret, "cost": cost, "strategy": strat}
    )


def run_backtest(panel: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    """Equal-weight portfolio of per-ticker threshold strategies vs. buy-and-hold.

    The first panel day of each ticker is dropped (its return is undefined).
    """
    cfg = cfg or BacktestConfig()
    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, grp in panel.groupby("ticker", sort=True):
        res = run_ticker(grp, cfg)
        per_ticker[str(ticker)] = res.iloc[1:]
    if not per_ticker:
        raise ValueError("empty panel")
    strat = pd.concat({t: r["strategy"] for t, r in per_ticker.items()}, axis=1).mean(axis=1)
    bench = pd.concat({t: r["ret"] for t, r in per_ticker.items()}, axis=1).mean(axis=1)
    exposure = pd.concat({t: r["position"].abs() for t, r in per_ticker.items()}, axis=1)
    trades = sum(int((r["position"].diff().fillna(0) != 0).sum()) for r in per_ticker.values())
    daily = pd.DataFrame(
        {
            "strategy": strat,
            "buy_hold": bench,
            "strategy_equity": (1 + strat).cumprod(),
            "buy_hold_equity": (1 + bench).cumprod(),
        }
    )
    m_strat = performance(strat, cfg.periods_per_year)
    # Hit rate = share of (ticker, day) positions whose direction matched the move.
    active = pd.concat([r[r["position"] != 0] for r in per_ticker.values()])
    m_strat["hit_rate"] = (
        float((active["position"] * active["ret"] > 0).mean()) if len(active) else float("nan")
    )
    m_strat["exposure"] = float(exposure.mean(axis=1).mean())
    m_strat["trades"] = float(trades)
    m_bench = performance(bench, cfg.periods_per_year)
    all_rets = pd.concat([r["ret"] for r in per_ticker.values()])
    m_bench["hit_rate"] = float((all_rets[all_rets != 0] > 0).mean())
    m_bench["exposure"] = 1.0
    m_bench["trades"] = float(len(per_ticker))
    metrics = pd.DataFrame({"strategy": m_strat, "buy_hold": m_bench})
    return BacktestResult(daily=daily, metrics=metrics, per_ticker=per_ticker)
