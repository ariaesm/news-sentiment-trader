"""Tests for ticker tagging."""

from __future__ import annotations

from pathlib import Path

from news_sentiment_trader.models import NewsItem
from news_sentiment_trader.tagging import TickerTagger


def _tagger() -> TickerTagger:
    return TickerTagger({"BTC-USD": ["bitcoin", "BTC"], "AAPL": ["Apple", "AAPL"]})


def test_case_insensitive_words_and_case_sensitive_symbols() -> None:
    tagger = _tagger()
    assert tagger.find("BITCOIN rallies") == ("BTC-USD",)
    assert tagger.find("BTC breaks 100k") == ("BTC-USD",)
    # Lowercase "btc" is not a symbol match; "apple" (word) is case-insensitive.
    assert tagger.find("btc fans eat an apple") == ("AAPL",)


def test_whole_word_matching() -> None:
    tagger = _tagger()
    assert tagger.find("Pineapple prices rise") == ()
    assert tagger.find("Applesauce") == ()
    assert tagger.find("Shares of $AAPL rise") == ("AAPL",)


def test_multiple_tickers_and_merge(sample_items: list[NewsItem]) -> None:
    tagger = _tagger()
    item = sample_items[0].with_tickers(["MSFT"])
    tagged = tagger.tag(item)
    assert tagged.tickers == ("MSFT", "AAPL")
    assert tagger.tag(item, keep_existing=False).tickers == ("AAPL",)


def test_from_yaml_bundled_config(repo_root: Path, sample_items: list[NewsItem]) -> None:
    tagger = TickerTagger.from_yaml(repo_root / "config" / "tickers.yaml")
    assert set(tagger.tickers) == {"BTC-USD", "AAPL", "MSFT"}
    tagged = tagger.tag_all(sample_items)
    assert tagged[0].tickers == ("AAPL",)
    assert tagged[1].tickers == ("BTC-USD",)
    assert set(tagged[2].tickers) == {"AAPL", "MSFT"}
