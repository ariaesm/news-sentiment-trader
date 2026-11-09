"""news-sentiment-trader: does news sentiment lead price moves?

A research pipeline that scores financial news headlines (LLM or offline lexicon),
aggregates sentiment per asset per day, measures lead-lag relationships with
returns, and backtests a simple, look-ahead-free sentiment signal.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
