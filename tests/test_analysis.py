"""Tests for the statistical analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trader.analysis import (
    add_forward_returns,
    correlations,
    event_study,
    hit_rate,
    lead_lag,
)


def make_panel(n: int = 400, lead: int = 1, strength: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Synthetic single-ticker panel where sentiment predicts ret ``lead`` days later."""
    rng = np.random.default_rng(seed)
    sent = rng.normal(size=n)
    ret = rng.normal(scale=0.01, size=n)
    ret[lead:] += strength * 0.01 * sent[: n - lead]
    close = 100 * np.cumprod(1 + ret)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"date": dates, "ticker": "AAA", "close": close, "ret": ret, "sent_mean": sent}
    )


def test_forward_returns_align() -> None:
    panel = make_panel(10)
    out = add_forward_returns(panel, [1, 2])
    expected = panel["close"].iloc[2] / panel["close"].iloc[0] - 1
    assert out["fwd_ret_2"].iloc[0] == pytest.approx(expected)
    assert out["fwd_ret_1"].isna().sum() == 1
    with pytest.raises(ValueError):
        add_forward_returns(panel, [0])


def test_forward_returns_do_not_cross_tickers() -> None:
    a = make_panel(5)
    b = make_panel(5, seed=1).assign(ticker="BBB")
    out = add_forward_returns(pd.concat([a, b]), [1])
    assert out.groupby("ticker")["fwd_ret_1"].apply(lambda s: s.isna().sum()).tolist() == [1, 1]


def test_correlations_detect_planted_signal() -> None:
    res = correlations(make_panel(), horizons=[1])
    row = res[res["ticker"] == "ALL"].iloc[0]
    assert row["pearson"] > 0.4 and row["pearson_p"] < 1e-6
    assert row["spearman"] > 0.4


def test_correlations_null_signal_is_small() -> None:
    res = correlations(make_panel(strength=0.0), horizons=[1])
    assert abs(res.iloc[0]["pearson"]) < 0.15


def test_lead_lag_peaks_at_planted_lag() -> None:
    ll = lead_lag(make_panel(lead=2))
    pooled = ll[ll["ticker"] == "ALL"].set_index("lag")["pearson"]
    assert pooled.idxmax() == 2
    assert set(pooled.index) == set(range(-5, 6))


def test_event_study_separates_extremes() -> None:
    res = event_study(make_panel(), quantile=0.9, window=3)
    table = res.table.set_index("day")
    assert res.n_pos > 0 and res.n_neg > 0
    assert table.loc[1, "pos"] > 0 > table.loc[1, "neg"]
    assert table.loc[1, "pos_lo"] < table.loc[1, "pos"] < table.loc[1, "pos_hi"]
    assert table.loc[0, "pos"] == 0.0


def test_hit_rate_and_binomial_pvalue() -> None:
    hr = hit_rate(make_panel())
    assert hr["hit_rate"] > 0.6 and hr["p_value"] < 1e-3
    null = hit_rate(make_panel(strength=0.0))
    assert 0.4 < null["hit_rate"] < 0.6


def test_hit_rate_empty() -> None:
    panel = make_panel(20).assign(sent_mean=np.nan)
    assert hit_rate(panel)["n"] == 0
