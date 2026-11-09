"""Markdown + PNG research report."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from news_sentiment_trader.analysis import (
    POOLED,
    EventStudyResult,
    correlations,
    event_study,
    hit_rate,
    lead_lag,
)
from news_sentiment_trader.backtest import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)

# Validated categorical slots (blue, orange, aqua) and recessive inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(rows: int = 1, height: float = 3.4, **kw: object) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(rows, 1, figsize=(8, height), squeeze=False, **kw)  # type: ignore[arg-type]
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes.ravel()


def plot_sentiment_vs_price(panel: pd.DataFrame, path: Path, signal: str = "sent_decay") -> None:
    """Per ticker: close price (top) and decayed sentiment (bottom), shared x-axis."""
    tickers = sorted(panel["ticker"].unique())
    fig, axes = _fig(2 * len(tickers), height=2.6 * len(tickers), sharex=True)
    for i, ticker in enumerate(tickers):
        g = panel[panel["ticker"] == ticker].sort_values("date")
        ax_p, ax_s = axes[2 * i], axes[2 * i + 1]
        ax_p.plot(g["date"], g["close"], color=BLUE, linewidth=1.6)
        ax_p.set_title(f"{ticker}: close", loc="left", fontsize=9, color=INK)
        s = g[signal]
        ax_s.fill_between(g["date"], 0, s, where=s >= 0, color=BLUE, alpha=0.35, linewidth=0)
        ax_s.fill_between(g["date"], 0, s, where=s < 0, color=ORANGE, alpha=0.35, linewidth=0)
        ax_s.plot(g["date"], s, color=INK_2, linewidth=0.8)
        ax_s.axhline(0, color=MUTED, linewidth=0.8)
        ax_s.set_ylim(-1, 1)
        ax_s.set_title(f"{ticker}: sentiment ({signal})", loc="left", fontsize=9, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lead_lag(ll: pd.DataFrame, path: Path) -> None:
    """Pooled Pearson correlation by lag with an approximate 95% null band."""
    pooled = ll[ll["ticker"] == POOLED].sort_values("lag")
    fig, (ax,) = _fig()
    colors = [BLUE if lag > 0 else MUTED for lag in pooled["lag"]]
    ax.bar(pooled["lag"], pooled["pearson"], color=colors, width=0.6)
    n = float(pooled["n"].median()) if len(pooled) else 0.0
    if n > 3:
        band = 1.96 / math.sqrt(n)
        for y in (band, -band):
            ax.axhline(y, color=INK_2, linewidth=0.8, linestyle="--")
        ax.text(pooled["lag"].max() + 0.4, band, "95% null", fontsize=7, color=INK_2, va="bottom")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(pooled["lag"])
    ax.set_xlabel(
        "lag k (days): corr(sentiment[t], return[t+k]); k > 0 = sentiment leads",
        fontsize=8,
        color=INK_2,
    )
    ax.set_ylabel("Pearson r", fontsize=8, color=INK_2)
    ax.set_title("Lead-lag correlation (all tickers pooled)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity(bt: BacktestResult, path: Path) -> None:
    """Strategy vs equal-weight buy-and-hold equity curves, direct-labelled."""
    fig, (ax,) = _fig(height=3.6)
    d = bt.daily
    for col, label, color in (
        ("strategy_equity", "Sentiment strategy", BLUE),
        ("buy_hold_equity", "Buy & hold (equal weight)", ORANGE),
    ):
        ax.plot(d.index, d[col], color=color, linewidth=2, label=label)
        ax.annotate(
            f"{label}  {d[col].iloc[-1]:.2f}x",
            (d.index[-1], d[col].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK,
            va="center",
        )
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylabel("growth of 1", fontsize=8, color=INK_2)
    ax.set_title("Equity curve (after costs)", loc="left", fontsize=10, color=INK)
    span = d.index[-1] - d.index[0]
    ax.set_xlim(d.index[0] - span * 0.01, d.index[-1] + span * 0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_event_study(es: EventStudyResult, path: Path) -> None:
    """Mean cumulative return after extreme days with confidence bands."""
    fig, (ax,) = _fig()
    t = es.table
    for key, label, color, n in (
        ("pos", "strongly positive days", BLUE, es.n_pos),
        ("neg", "strongly negative days", ORANGE, es.n_neg),
    ):
        ax.plot(
            t["day"],
            t[key] * 100,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{label} (n={n})",
        )
        ax.fill_between(
            t["day"],
            t[f"{key}_lo"] * 100,
            t[f"{key}_hi"] * 100,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("trading days after the news day", fontsize=8, color=INK_2)
    ax.set_ylabel("mean cumulative return (%)", fontsize=8, color=INK_2)
    ax.set_title("Event study (95% CI)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = ".4f", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate needed)."""
    if index:
        df = df.reset_index()

    def fmt(v: object) -> str:
        if isinstance(v, float | np.floating):
            return "n/a" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


@dataclass(slots=True)
class ReportResult:
    """Paths written and the headline numbers (also used by the CLI)."""

    markdown: Path
    charts: dict[str, Path]
    numbers: dict[str, float] = field(default_factory=dict)


def build_report(
    scored: pd.DataFrame,
    panel: pd.DataFrame,
    out_dir: str | Path,
    bt_config: BacktestConfig | None = None,
    signal: str = "sent_mean",
    scorer_name: str = "unknown",
    note: str = "",
) -> ReportResult:
    """Run analysis + backtest on ``panel`` and write ``report.md`` and PNG charts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = bt_config or BacktestConfig(signal=signal)

    corr = correlations(panel, signal, horizons=(1, 5))
    ll = lead_lag(panel, signal)
    es = event_study(panel, signal)
    hr = hit_rate(panel, signal)
    bt = run_backtest(panel, cfg)

    charts = {
        "sentiment_vs_price": out / "sentiment_vs_price.png",
        "lead_lag": out / "lead_lag.png",
        "equity_curve": out / "equity_curve.png",
        "event_study": out / "event_study.png",
    }
    plot_sentiment_vs_price(panel, charts["sentiment_vs_price"])
    plot_lead_lag(ll, charts["lead_lag"])
    plot_equity(bt, charts["equity_curve"])
    plot_event_study(es, charts["event_study"])

    pooled1 = corr[(corr["ticker"] == POOLED) & (corr["horizon"] == 1)].iloc[0]
    numbers = {
        "headlines": float(scored["id"].nunique()) if "id" in scored else float(len(scored)),
        "news_days": float(panel[signal].notna().sum()),
        "pearson_1d": float(pooled1["pearson"]),
        "pearson_1d_p": float(pooled1["pearson_p"]),
        "spearman_1d": float(pooled1["spearman"]),
        "hit_rate_1d": float(hr["hit_rate"]),
        "hit_rate_p": float(hr["p_value"]),
        **{f"strategy_{k}": float(v) for k, v in bt.metrics["strategy"].items()},
        **{f"buy_hold_{k}": float(v) for k, v in bt.metrics["buy_hold"].items()},
    }

    ll_wide = ll.pivot(index="lag", columns="ticker", values="pearson").reset_index()
    metrics = bt.metrics.copy()
    metrics.index.name = "metric"
    start, end = panel["date"].min().date(), panel["date"].max().date()
    short = "long/short" if cfg.allow_short else "long/flat"
    lines = [
        "# News sentiment research report",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by news-sentiment-trader._",
        "",
    ]
    if note:
        lines += [f"> **Note:** {note}", ""]
    lines += [
        "## Setup",
        "",
        f"- Scorer: `{scorer_name}`; signal column: `{signal}`",
        f"- Tickers: {', '.join(sorted(panel['ticker'].unique()))}; period {start} to {end}",
        f"- Headline-ticker rows scored: {len(scored)}; ticker-days with news: "
        f"{int(numbers['news_days'])}",
        f"- Backtest: {short}, threshold ±{cfg.threshold}, costs {cfg.cost_bps} bps per unit "
        "turnover, equal-weight across tickers, position on day t uses only the signal "
        "known at the close of t-1.",
        "",
        "![Sentiment vs price](sentiment_vs_price.png)",
        "",
        "## Does sentiment predict forward returns?",
        "",
        "Correlation of daily sentiment with the k-day forward return "
        "(news days only; 5-day windows overlap, so their p-values are optimistic).",
        "",
        md_table(corr),
        "",
        f"Directional hit rate (next day): **{hr['hit_rate']:.1%}** over {hr['n']} news days "
        f"(one-sided binomial p = {hr['p_value']:.3g} vs. 50%).",
        "",
        "## Lead-lag",
        "",
        "Pearson correlation between sentiment on day t and the return on day t+k.",
        "",
        "![Lead-lag](lead_lag.png)",
        "",
        md_table(ll_wide),
        "",
        "## Event study",
        "",
        f"Strongly positive days: signal ≥ {es.pos_threshold:.3f} (n={es.n_pos}); strongly "
        f"negative days: signal ≤ {es.neg_threshold:.3f} (n={es.n_neg}). Thresholds are "
        "in-sample quantiles, so this section is descriptive.",
        "",
        "![Event study](event_study.png)",
        "",
        md_table(es.table),
        "",
        "## Backtest",
        "",
        "![Equity curve](equity_curve.png)",
        "",
        md_table(metrics, index=True),
        "",
        "## Caveats",
        "",
        "- One dataset, one parameter set: no out-of-sample period and no correction for "
        "the many lags/horizons tested.",
        "- Costs are a flat bps charge; no slippage, borrow costs, or capacity limits.",
        "- Not financial advice. This is a research tool.",
        "",
    ]
    md_path = out / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    bt.daily.to_csv(out / "backtest_daily.csv")
    corr.to_csv(out / "correlations.csv", index=False)
    ll.to_csv(out / "lead_lag.csv", index=False)
    return ReportResult(markdown=md_path, charts=charts, numbers=numbers)
