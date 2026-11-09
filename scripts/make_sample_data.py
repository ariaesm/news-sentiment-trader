"""Generate the SYNTHETIC sample dataset in ``data/``.

Everything produced here is fake: prices are a random walk and headlines are
filled-in templates. A weak relationship is planted on purpose so that the
pipeline has something to find:

    news tone on trading day D  ~  latent shock z[D] + noise
    return on day D+1           =  drift + vol * (RHO * z[D] + sqrt(1 - RHO^2) * eps)

Headlines for day D are timestamped between the previous trading day's 20:00
UTC cutoff and D's 20:00 UTC cutoff, so after aggregation they land on D.

Usage::

    python scripts/make_sample_data.py [--seed 42] [--days 250] [--out data]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RHO = 0.3  # correlation between the latent shock and the next-day return
NEWS_RATE = 0.6  # Poisson mean of headlines per ticker per day
CUTOFF_HOUR = 20

ASSETS: dict[str, dict[str, object]] = {
    "BTC-USD": {"names": ["Bitcoin", "BTC"], "start": 42000.0, "vol": 0.032, "drift": 0.0008},
    "AAPL": {"names": ["Apple"], "start": 185.0, "vol": 0.015, "drift": 0.0004},
    "MSFT": {"names": ["Microsoft"], "start": 375.0, "vol": 0.014, "drift": 0.0004},
}

POSITIVE = [
    "{name} shares jump {pct}% after earnings beat estimates",
    "{name} rallies as analysts upgrade outlook",
    "{name} surges to record high on strong demand",
    "Analysts turn bullish on {name} after upbeat guidance",
    "{name} soars {pct}% as investors cheer growth",
    "{name} gains on robust quarterly results",
    "{name} climbs after regulators grant approval",
    "{name} raises guidance, shares rise",
    "Inflows into {name} products hit record",
    "{name} rebounds strongly as momentum returns",
    "Broker raises {name} price target to new high",
    "{name} announces buyback; stock jumps",
    "{name} partnership sparks optimism among investors",
    "{name} beats estimates on resilient sales",
    "{name} extends rally for third day",
]
NEGATIVE = [
    "{name} shares plunge {pct}% after earnings miss",
    "{name} tumbles as analysts downgrade the stock",
    "{name} slumps amid regulatory probe",
    "Investors flee {name} as outflows accelerate",
    "{name} falls sharply on weak guidance",
    "{name} drops after outage disrupts users",
    "{name} slides as lawsuit fears mount",
    "{name} cuts guidance, warns of headwinds",
    "Selloff hits {name} as concerns grow",
    "{name} sinks {pct}% on disappointing results",
    "Antitrust scrutiny weighs on {name}",
    "{name} misses estimates as demand slows",
    "{name} hit by security breach, shares decline",
    "Bearish analysts see more losses for {name}",
    "{name} delays product launch, stock falls",
]
NEUTRAL = [
    "{name} to hold annual shareholder meeting",
    "{name} schedules quarterly earnings call",
    "What to watch for {name} this week",
    "{name} appoints new board member",
    "{name} trading volume in focus ahead of data",
    "{name} files routine regulatory paperwork",
    "Explainer: how {name} reports its revenue",
    "{name} unchanged in early trading",
    "{name} executives speak at industry conference",
    "Options traders eye {name} expiry",
]
SOURCES = ["Synthetic Wire", "Synthetic Markets Daily", "Synthetic Finance Desk"]


def make_prices(rng: np.random.Generator, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Return (long price frame, latent shocks per ticker)."""
    n = len(dates)
    rows: list[dict[str, object]] = []
    shocks: dict[str, np.ndarray] = {}
    for ticker, spec in ASSETS.items():
        z = rng.normal(size=n)
        eps = rng.normal(size=n)
        ret = np.zeros(n)
        ret[1:] = float(spec["drift"]) + float(spec["vol"]) * (
            RHO * z[:-1] + np.sqrt(1 - RHO**2) * eps[1:]
        )
        close = float(spec["start"]) * np.cumprod(1 + ret)
        shocks[ticker] = z
        for d, c in zip(dates, close, strict=True):
            rows.append({"date": d.date().isoformat(), "ticker": ticker, "close": round(c, 4)})
    return pd.DataFrame(rows), shocks


def make_news(
    rng: np.random.Generator, dates: pd.DatetimeIndex, shocks: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    """Return synthetic headline records whose tone tracks the latent shocks."""
    items: list[dict[str, object]] = []
    for i, day in enumerate(dates):
        window_end = datetime(day.year, day.month, day.day, CUTOFF_HOUR, tzinfo=UTC)
        prev = dates[i - 1] if i > 0 else day - pd.tseries.offsets.BDay(1)
        window_start = datetime(prev.year, prev.month, prev.day, CUTOFF_HOUR, tzinfo=UTC)
        span = (window_end - window_start).total_seconds()
        for ticker, spec in ASSETS.items():
            for _ in range(rng.poisson(NEWS_RATE)):
                tone = shocks[ticker][i] + rng.normal(scale=0.9)
                pool = POSITIVE if tone > 0.45 else NEGATIVE if tone < -0.45 else NEUTRAL
                template = pool[rng.integers(len(pool))]
                names = list(spec["names"])  # type: ignore[call-overload]
                title = template.format(
                    name=names[rng.integers(len(names))], pct=round(rng.uniform(1.0, 9.5), 1)
                )
                title = title[0].upper() + title[1:]
                ts = window_start + timedelta(seconds=float(rng.uniform(0, span - 1)))
                items.append(
                    {
                        "id": f"syn-{len(items):05d}",
                        "timestamp": ts.replace(microsecond=0).isoformat(),
                        "source": SOURCES[rng.integers(len(SOURCES))],
                        "title": title,
                        "summary": "",
                    }
                )
    items.sort(key=lambda r: str(r["timestamp"]))
    return items


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=250)
    parser.add_argument("--start", default="2025-01-02")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range(args.start, periods=args.days)
    prices, shocks = make_prices(rng, dates)
    news = make_news(rng, dates, shocks)

    args.out.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out / "sample_prices.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    payload = {
        "_note": (
            "SYNTHETIC DATA generated by scripts/make_sample_data.py "
            f"(seed={args.seed}). Not real news; for demonstration only."
        ),
        "items": news,
    }
    (args.out / "sample_news.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(prices)} price rows and {len(news)} headlines to {args.out}/")


if __name__ == "__main__":
    main()
