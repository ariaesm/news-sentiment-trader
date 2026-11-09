"""Tests for the backtest, including explicit look-ahead checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trader.backtest import (
    BacktestConfig,
    max_drawdown,
    performance,
    positions,
    run_backtest,
    run_ticker,
)


def _panel(sent: np.ndarray, ret: np.ndarray, ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(ret))
    close = 100 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {"date": dates, "ticker": ticker, "close": close, "ret": ret, "sent_mean": sent}
    )


def test_positions_are_lagged_one_day() -> None:
    sig = pd.Series([0.5, -0.5, 0.1, np.nan, 0.9])
    pos = positions(sig, threshold=0.2)
    assert pos.tolist() == [0.0, 1.0, -1.0, 0.0, 0.0]
    assert positions(sig, 0.2, allow_short=False).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_no_lookahead_future_signal_does_not_change_past() -> None:
    """Perturbing the signal from day k onward must not alter positions before k+1."""
    rng = np.random.default_rng(1)
    sent = rng.normal(size=100)
    ret = rng.normal(scale=0.01, size=100)
    base = run_ticker(_panel(sent, ret), BacktestConfig(threshold=0.1))
    k = 60
    sent2 = sent.copy()
    sent2[k:] = -sent2[k:] * 10
    alt = run_ticker(_panel(sent2, ret), BacktestConfig(threshold=0.1))
    pd.testing.assert_frame_equal(
        base.iloc[: k + 1][["position", "strategy"]], alt.iloc[: k + 1][["position", "strategy"]]
    )
    assert not base["position"].iloc[k + 1 :].equals(alt["position"].iloc[k + 1 :])


def test_same_day_return_signal_is_not_exploitable() -> None:
    """A 'signal' equal to the same-day return is perfect foresight if misaligned.

    With correct lagging it carries no edge on i.i.d. returns.
    """
    rng = np.random.default_rng(7)
    ret = rng.normal(scale=0.01, size=2000)
    cfg = BacktestConfig(threshold=0.0, cost_bps=0.0)
    res = run_backtest(_panel(ret.copy(), ret), cfg)
    assert abs(res.metrics.loc["sharpe", "strategy"]) < 1.0
    # Whereas a genuinely leading signal (yesterday's value of tomorrow's return) works.
    leading = np.roll(ret, -1)
    res2 = run_backtest(_panel(leading, ret), cfg)
    assert res2.metrics.loc["sharpe", "strategy"] > 10


def test_costs_reduce_returns() -> None:
    rng = np.random.default_rng(3)
    sent = rng.normal(size=300)
    ret = rng.normal(scale=0.01, size=300)
    free = run_backtest(_panel(sent, ret), BacktestConfig(cost_bps=0.0))
    costly = run_backtest(_panel(sent, ret), BacktestConfig(cost_bps=20.0))
    assert (
        costly.metrics.loc["total_return", "strategy"]
        < free.metrics.loc["total_return", "strategy"]
    )
    assert costly.metrics.loc["trades", "strategy"] > 0


def test_equal_weight_portfolio_and_benchmark() -> None:
    ret_a = np.full(10, 0.01)
    ret_b = np.full(10, -0.01)
    panel = pd.concat([_panel(np.ones(10), ret_a, "A"), _panel(np.ones(10), ret_b, "B")])
    res = run_backtest(panel, BacktestConfig(cost_bps=0.0))
    assert res.daily["buy_hold"].abs().max() == pytest.approx(0.0)
    assert set(res.per_ticker) == {"A", "B"}
    assert len(res.daily) == 9


def test_metrics_math() -> None:
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(eq) == pytest.approx(-0.25)
    perf = performance(pd.Series([0.01, -0.02, 0.0, 0.03]), periods_per_year=252)
    assert perf["total_return"] == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    assert perf["hit_rate"] == pytest.approx(2 / 3)
    assert np.isnan(performance(pd.Series([], dtype=float))["sharpe"])
