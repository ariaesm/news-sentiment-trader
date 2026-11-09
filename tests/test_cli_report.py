"""End-to-end tests for the CLI and report generator on the synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from news_sentiment_trader.cli import main
from news_sentiment_trader.report import md_table


@pytest.fixture
def in_repo(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(repo_root)
    return repo_root


def test_run_all_end_to_end(in_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "rep"
    code = main(
        [
            "run-all",
            "--news",
            "data/sample_news.json",
            "--prices",
            "data/sample_prices.csv",
            "--scorer",
            "lexicon",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    for name in (
        "report.md",
        "equity_curve.png",
        "lead_lag.png",
        "sentiment_vs_price.png",
        "event_study.png",
        "scored.csv",
        "summary.json",
    ):
        assert (out / name).exists(), name
    report = (out / "report.md").read_text()
    assert "SYNTHETIC" in report and "Not financial advice" in report
    assert "![Equity curve](equity_curve.png)" in report
    summary = json.loads((out / "summary.json").read_text())
    # The synthetic data has a planted positive relationship; the lexicon should find it.
    assert summary["pearson_1d"] > 0
    assert "headlines=" in capsys.readouterr().out


def test_score_analyze_backtest_report(in_repo: Path, tmp_path: Path) -> None:
    scored = tmp_path / "scored.csv"
    assert main(["score", "--news", "data/sample_news.json", "--out", str(scored)]) == 0
    df = pd.read_csv(scored)
    assert set(df["ticker"]) == {"AAPL", "BTC-USD", "MSFT"}
    common = ["--scored", str(scored), "--prices", "data/sample_prices.csv"]
    assert main(["analyze", *common, "--out", str(tmp_path / "an")]) == 0
    assert (tmp_path / "an" / "lead_lag.csv").exists()
    assert main(["backtest", *common, "--long-only", "--out", str(tmp_path / "bt.csv")]) == 0
    assert main(["report", *common, "--signal", "sent_decay", "--out", str(tmp_path / "r")]) == 0
    assert "sent_decay" in (tmp_path / "r" / "report.md").read_text()


def test_cli_reports_missing_file(capsys: pytest.CaptureFixture) -> None:
    assert main(["score", "--news", "does/not/exist.json"]) == 2
    assert "error:" in capsys.readouterr().err


def test_md_table_formats_nan() -> None:
    table = md_table(pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", "y"]}), floatfmt=".1f")
    assert table.splitlines()[0] == "| a | b |"
    assert "| n/a | y |" in table
