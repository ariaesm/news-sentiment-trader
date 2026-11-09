"""Command-line interface: ``nst score | analyze | backtest | report | run-all``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from news_sentiment_trader import __version__
from news_sentiment_trader.aggregate import (
    build_panel,
    load_prices,
    load_scored,
    save_scored,
    scored_frame,
)
from news_sentiment_trader.analysis import POOLED, correlations, event_study, hit_rate, lead_lag
from news_sentiment_trader.backtest import BacktestConfig, run_backtest
from news_sentiment_trader.ingest import deduplicate, load_news
from news_sentiment_trader.report import build_report, md_table
from news_sentiment_trader.scorers import LexiconScorer, LLMConfig, LLMScorer, SentimentScorer
from news_sentiment_trader.tagging import TickerTagger

DEFAULT_TICKERS = Path("config/tickers.yaml")


def make_scorer(args: argparse.Namespace) -> SentimentScorer:
    """Instantiate the scorer selected on the command line."""
    if args.scorer == "lexicon":
        return LexiconScorer()
    cfg = LLMConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        cache_dir=Path(args.cache_dir),
    )
    return LLMScorer(cfg)


def _news_note(path: str) -> str:
    """Return the ``_note`` field of a JSON news file, if any (used to flag synthetic data)."""
    if not path.endswith(".json") or not Path(path).exists():
        return ""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("_note", "")) if isinstance(data, dict) else ""


def score_news(args: argparse.Namespace) -> pd.DataFrame:
    """Load, deduplicate, tag and score news; return the scored frame."""
    items = deduplicate(load_news(args.news))
    tagger = TickerTagger.from_yaml(args.tickers)
    items = [i for i in tagger.tag_all(items) if i.tickers]
    scorer = make_scorer(args)
    results = scorer.score_texts([i.text for i in items])
    return scored_frame(items, results)


def _panel(args: argparse.Namespace, scored: pd.DataFrame) -> pd.DataFrame:
    return build_panel(scored, load_prices(args.prices), args.cutoff_hour, args.halflife)


def _bt_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        signal=args.signal,
        threshold=args.threshold,
        allow_short=not args.long_only,
        cost_bps=args.cost_bps,
        periods_per_year=args.periods_per_year,
    )


# -- commands -------------------------------------------------------------------
def cmd_score(args: argparse.Namespace) -> int:
    """``nst score``: write scored headlines to CSV."""
    scored = score_news(args)
    save_scored(scored, args.out)
    counts = scored["label"].value_counts().to_dict()
    print(
        f"scored {scored['id'].nunique()} headlines ({len(scored)} headline-ticker rows) "
        f"-> {args.out}  labels={counts}"
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """``nst analyze``: print correlations, lead-lag, event study and hit rate."""
    panel = _panel(args, load_scored(args.scored))
    corr = correlations(panel, args.signal, horizons=args.horizons)
    ll = lead_lag(panel, args.signal)
    es = event_study(panel, args.signal)
    hr = hit_rate(panel, args.signal)
    print("## Correlation with forward returns\n" + md_table(corr) + "\n")
    pooled = ll[ll["ticker"] == POOLED][["lag", "pearson", "spearman", "n"]]
    print("## Lead-lag (pooled)\n" + md_table(pooled) + "\n")
    print(f"## Event study (n_pos={es.n_pos}, n_neg={es.n_neg})\n" + md_table(es.table) + "\n")
    print(f"Hit rate (1d): {hr['hit_rate']:.3f} over {hr['n']} days, p={hr['p_value']:.3g}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out / "panel.csv", index=False)
        corr.to_csv(out / "correlations.csv", index=False)
        ll.to_csv(out / "lead_lag.csv", index=False)
        es.table.to_csv(out / "event_study.csv", index=False)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """``nst backtest``: print strategy vs buy-and-hold metrics."""
    panel = _panel(args, load_scored(args.scored))
    bt = run_backtest(panel, _bt_config(args))
    print(md_table(bt.metrics, index=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        bt.daily.to_csv(args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """``nst report``: write the Markdown report and charts."""
    scored = load_scored(args.scored)
    res = build_report(
        scored,
        _panel(args, scored),
        args.out,
        _bt_config(args),
        args.signal,
        scorer_name=args.scorer_name,
        note=args.note,
    )
    print(f"report written to {res.markdown}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """``nst run-all``: score -> aggregate -> analyze -> backtest -> report."""
    out = Path(args.out)
    scored = score_news(args)
    save_scored(scored, out / "scored.csv")
    res = build_report(
        scored,
        _panel(args, scored),
        out,
        _bt_config(args),
        args.signal,
        scorer_name=args.scorer,
        note=args.note or _news_note(args.news),
    )
    (out / "summary.json").write_text(json.dumps(res.numbers, indent=2) + "\n")
    n = res.numbers
    print(
        f"headlines={int(n['headlines'])}  pearson_1d={n['pearson_1d']:+.3f} "
        f"(p={n['pearson_1d_p']:.3g})  hit_rate={n['hit_rate_1d']:.1%}"
    )
    print(
        f"strategy: total={n['strategy_total_return']:+.1%} sharpe={n['strategy_sharpe']:.2f} "
        f"maxDD={n['strategy_max_drawdown']:.1%} | buy&hold: "
        f"total={n['buy_hold_total_return']:+.1%} sharpe={n['buy_hold_sharpe']:.2f} "
        f"maxDD={n['buy_hold_max_drawdown']:.1%}"
    )
    print(f"report written to {res.markdown}")
    return 0


# -- parser ---------------------------------------------------------------------
def _add_scoring(p: argparse.ArgumentParser) -> None:
    p.add_argument("--news", required=True, help="news file (.json/.csv/.xml) or RSS URL")
    p.add_argument("--tickers", default=str(DEFAULT_TICKERS), help="ticker keyword YAML")
    p.add_argument("--scorer", choices=["lexicon", "llm"], default="lexicon")
    p.add_argument("--model", default=None, help="LLM model (default: $NST_LLM_MODEL)")
    p.add_argument("--base-url", default=None, help="LLM base URL (default: $NST_LLM_BASE_URL)")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--cache-dir", default=".cache/llm")


def _add_panel(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prices", required=True, help="daily close CSV (long or wide)")
    p.add_argument("--signal", default="sent_mean", choices=["sent_mean", "sent_decay"])
    p.add_argument(
        "--cutoff-hour",
        type=float,
        default=20.0,
        help="UTC hour after which news counts for the next trading day",
    )
    p.add_argument("--halflife", type=float, default=3.0, help="decay half-life (trading days)")


def _add_backtest(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--long-only", action="store_true", help="flat instead of short")
    p.add_argument("--periods-per-year", type=int, default=252)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="nst", description="News sentiment research pipeline (not financial advice)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="tag and score headlines")
    _add_scoring(p)
    p.add_argument("--out", default="outputs/scored.csv")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze", help="correlation, lead-lag, event study, hit rate")
    p.add_argument("--scored", required=True)
    _add_panel(p)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--out", default=None, help="optional directory for CSV outputs")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("backtest", help="backtest the threshold strategy")
    p.add_argument("--scored", required=True)
    _add_panel(p)
    _add_backtest(p)
    p.add_argument("--out", default=None, help="optional CSV for daily returns")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="write Markdown report with charts")
    p.add_argument("--scored", required=True)
    _add_panel(p)
    _add_backtest(p)
    p.add_argument("--scorer-name", default="unknown")
    p.add_argument("--note", default="", help="note shown at the top of the report")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run-all", help="score + analyze + backtest + report")
    _add_scoring(p)
    _add_panel(p)
    _add_backtest(p)
    p.add_argument("--note", default="", help="note shown at the top of the report")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_run_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``nst`` console script."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
