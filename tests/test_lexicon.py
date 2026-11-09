"""Tests for the offline lexicon scorer."""

from __future__ import annotations

import pytest

from news_sentiment_trader.scorers import LexiconScorer, SentimentScorer
from news_sentiment_trader.scorers.lexicon import tokenize


@pytest.fixture
def scorer() -> LexiconScorer:
    return LexiconScorer()


def test_implements_protocol(scorer: LexiconScorer) -> None:
    assert isinstance(scorer, SentimentScorer)


def test_polarity(scorer: LexiconScorer) -> None:
    pos = scorer.score_text("Apple beats estimates on strong iPhone demand")
    neg = scorer.score_text("Bitcoin plunges as exchange hack sparks selloff")
    neutral = scorer.score_text("Microsoft to hold annual shareholder meeting")
    assert pos.label == "positive" and pos.score > 0.5
    assert neg.label == "negative" and neg.score < -0.5
    assert neutral.score == 0.0 and neutral.label == "neutral"


def test_scores_are_bounded(scorer: LexiconScorer) -> None:
    text = " ".join(["surge record rally soars bullish"] * 20)
    assert -1.0 <= scorer.score_text(text).score <= 1.0
    assert scorer.score_text(text).score > 0.99


def test_negation_flips_sign(scorer: LexiconScorer) -> None:
    plain = scorer.score_text("Outlook is strong").score
    negated = scorer.score_text("Outlook is not strong").score
    assert plain > 0 > negated
    assert abs(negated) < abs(plain)  # negated positives are softened


def test_negation_window_expires(scorer: LexiconScorer) -> None:
    # "not" is too far from "surge" to affect it.
    assert scorer.score_text("not that anyone expected the big surge").score > 0


def test_intensifiers_pre_and_post(scorer: LexiconScorer) -> None:
    base = scorer.score_text("Shares fall").score
    assert scorer.score_text("Shares fall sharply").score < base
    assert scorer.score_text("Shares sharply fall").score < base
    assert scorer.score_text("Shares fall slightly").score > base


def test_phrases_take_priority(scorer: LexiconScorer) -> None:
    res = scorer.score_text("Tesla issues profit warning")
    assert res.score < 0
    assert "profit warning" in res.reason
    assert scorer.score_text("Bitcoin hits record low").score < 0


def test_tokenize_keeps_hyphens() -> None:
    assert tokenize("All-time HIGH, isn't it?") == ["all-time", "high", "isn't", "it"]


def test_score_texts_preserves_order(scorer: LexiconScorer) -> None:
    out = scorer.score_texts(["stocks surge", "stocks crash", "stocks"])
    assert [r.label for r in out] == ["positive", "negative", "neutral"]
