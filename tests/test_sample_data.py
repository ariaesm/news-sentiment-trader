"""Sanity checks on the bundled synthetic dataset and its generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from news_sentiment_trader.aggregate import load_prices


def test_bundled_data_is_labelled_synthetic(repo_root: Path) -> None:
    payload = json.loads((repo_root / "data" / "sample_news.json").read_text())
    assert "SYNTHETIC" in payload["_note"]
    assert 200 <= len(payload["items"]) <= 1000
    assert all(i["source"].startswith("Synthetic") for i in payload["items"])
    prices = load_prices(repo_root / "data" / "sample_prices.csv")
    assert list(prices.columns) == ["AAPL", "BTC-USD", "MSFT"]
    assert len(prices) == 250


def test_generator_is_deterministic(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "scripts" / "make_sample_data.py"
    subprocess.run(
        [sys.executable, str(script), "--seed", "42", "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for name in ("sample_news.json", "sample_prices.csv"):
        assert (tmp_path / name).read_text() == (repo_root / "data" / name).read_text()
