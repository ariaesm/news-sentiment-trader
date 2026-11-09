"""Sentiment scorers behind a common :class:`SentimentScorer` protocol."""

from news_sentiment_trader.scorers.base import SentimentScorer, score_items
from news_sentiment_trader.scorers.lexicon import LexiconScorer

__all__ = ["LexiconScorer", "SentimentScorer", "score_items"]
