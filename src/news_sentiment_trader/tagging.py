"""Keyword-based ticker / entity tagging driven by a YAML config."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from news_sentiment_trader.models import NewsItem


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern: re.Pattern[str]
    ticker: str


class TickerTagger:
    """Assign tickers to news items by matching configurable keywords.

    Each keyword is matched as a whole word. All-uppercase keywords (e.g. ``"BTC"``,
    ``"AAPL"``) are matched case-sensitively so that ordinary words are not
    mistaken for symbols; other keywords are case-insensitive.

    Example config::

        tickers:
          BTC-USD: [bitcoin, BTC]
          AAPL: [Apple, iPhone, AAPL]
    """

    def __init__(self, mapping: Mapping[str, Iterable[str]]) -> None:
        self.mapping: dict[str, list[str]] = {t: list(kws) for t, kws in mapping.items()}
        self._rules: list[_Rule] = []
        for ticker, keywords in self.mapping.items():
            for kw in keywords:
                kw = kw.strip()
                if not kw:
                    continue
                flags = 0 if (kw.isupper() and len(kw) <= 6) else re.IGNORECASE
                pat = re.compile(rf"(?<![\w$-]){re.escape(kw)}(?![\w-])", flags)
                self._rules.append(_Rule(pat, ticker))
                # Also match cashtags like $AAPL.
                if kw.isupper():
                    self._rules.append(_Rule(re.compile(rf"\${re.escape(kw)}\b"), ticker))

    @classmethod
    def from_yaml(cls, path: str | Path) -> TickerTagger:
        """Build a tagger from a YAML file with a top-level ``tickers`` mapping."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        mapping = data.get("tickers", data)
        if not isinstance(mapping, dict):
            raise ValueError("ticker config must be a mapping of ticker -> keywords")
        return cls(mapping)

    @property
    def tickers(self) -> list[str]:
        """All tickers known to the tagger, in config order."""
        return list(self.mapping)

    def find(self, text: str) -> tuple[str, ...]:
        """Return the tickers mentioned in ``text`` in first-seen config order."""
        hits: list[str] = []
        for rule in self._rules:
            if rule.ticker not in hits and rule.pattern.search(text):
                hits.append(rule.ticker)
        return tuple(hits)

    def tag(self, item: NewsItem, keep_existing: bool = True) -> NewsItem:
        """Return ``item`` with tickers found in its title and summary."""
        found = self.find(item.text)
        existing = item.tickers if keep_existing else ()
        merged = list(existing) + [t for t in found if t not in existing]
        return item.with_tickers(merged)

    def tag_all(self, items: Iterable[NewsItem], keep_existing: bool = True) -> list[NewsItem]:
        """Tag every item in ``items``."""
        return [self.tag(i, keep_existing=keep_existing) for i in items]
