"""Sentiment scorers behind a common :class:`SentimentScorer` protocol."""

from news_sentiment_trader.scorers.base import SentimentScorer, score_items
from news_sentiment_trader.scorers.lexicon import LexiconScorer
from news_sentiment_trader.scorers.llm import LLMConfig, LLMScorer

__all__ = ["LLMConfig", "LLMScorer", "LexiconScorer", "SentimentScorer", "score_items"]
