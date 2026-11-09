"""Offline finance-lexicon sentiment baseline.

A deliberately small, transparent scorer: token-level polarity weights, a few
multi-word phrases, negation that flips the next few sentiment words, and
intensifiers / diminishers that scale them. The raw sum is squashed into
[-1, 1] with ``x / sqrt(x^2 + alpha)`` (the same normalization VADER uses).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from news_sentiment_trader.models import SentimentResult, label_for

# Weights roughly in [-3, 3]; hand-written for financial headlines.
POSITIVE: dict[str, float] = {
    "beat": 2.0,
    "beats": 2.0,
    "surge": 2.5,
    "surges": 2.5,
    "soar": 2.5,
    "soars": 2.5,
    "rally": 2.0,
    "rallies": 2.0,
    "jump": 1.8,
    "jumps": 1.8,
    "gain": 1.5,
    "gains": 1.5,
    "rise": 1.2,
    "rises": 1.2,
    "climb": 1.3,
    "climbs": 1.3,
    "record": 1.5,
    "strong": 1.5,
    "growth": 1.3,
    "profit": 1.2,
    "profitable": 1.5,
    "upgrade": 2.0,
    "upgraded": 2.0,
    "upgrades": 2.0,
    "outperform": 1.8,
    "bullish": 2.2,
    "optimism": 1.5,
    "optimistic": 1.5,
    "approval": 1.5,
    "approved": 1.5,
    "approves": 1.5,
    "partnership": 1.0,
    "expands": 1.0,
    "expansion": 1.0,
    "boost": 1.5,
    "boosts": 1.5,
    "robust": 1.5,
    "upbeat": 1.8,
    "buyback": 1.3,
    "dividend": 0.8,
    "breakthrough": 2.0,
    "adoption": 1.2,
    "inflows": 1.5,
    "recovery": 1.3,
    "rebound": 1.5,
    "rebounds": 1.5,
    "tops": 1.5,
    "exceeds": 1.8,
    "raises": 1.0,
    "raised": 1.0,
    "wins": 1.5,
    "win": 1.3,
    "launch": 0.6,
    "launches": 0.6,
    "demand": 0.6,
    "momentum": 1.0,
    "resilient": 1.3,
    "accelerates": 1.3,
    "milestone": 1.2,
}
NEGATIVE: dict[str, float] = {
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "plunge": -2.8, "plunges": -2.8,
    "slump": -2.3, "slumps": -2.3, "crash": -3.0, "crashes": -3.0, "tumble": -2.3,
    "tumbles": -2.3, "fall": -1.3, "falls": -1.3, "drop": -1.5, "drops": -1.5,
    "decline": -1.4, "declines": -1.4, "slide": -1.5, "slides": -1.5, "loss": -1.5,
    "losses": -1.6, "weak": -1.5, "weaker": -1.5, "downgrade": -2.0, "downgraded": -2.0,
    "downgrades": -2.0, "underperform": -1.8, "bearish": -2.2, "lawsuit": -1.8,
    "sued": -1.8, "probe": -1.6, "investigation": -1.6, "fined": -1.6,
    "recall": -1.8, "recalls": -1.8, "hack": -2.5, "hacked": -2.5, "breach": -2.2,
    "fraud": -3.0, "bankruptcy": -3.0, "layoffs": -1.6, "cuts": -1.2, "cut": -1.2,
    "warns": -1.8, "warning": -1.6, "risk": -0.8, "risks": -0.8, "concerns": -1.2,
    "fears": -1.6, "selloff": -2.2, "outflows": -1.5, "delay": -1.2, "delays": -1.2,
    "delayed": -1.2, "ban": -2.0, "bans": -2.0, "crackdown": -2.0, "volatile": -0.8,
    "slowdown": -1.5, "shortage": -1.3, "disappointing": -2.0, "disappoints": -2.0,
    "halts": -1.8, "outage": -1.8, "scrutiny": -1.3, "antitrust": -1.4, "sink": -2.0,
    "sinks": -2.0, "lower": -0.6, "pessimism": -1.5, "headwinds": -1.4,
}  # fmt: skip
PHRASES: dict[str, float] = {
    "beats estimates": 2.5,
    "beat estimates": 2.5,
    "misses estimates": -2.5,
    "missed estimates": -2.5,
    "price target raised": 2.0,
    "price target cut": -2.0,
    "all-time high": 2.2,
    "record high": 2.2,
    "record low": -2.2,
    "record losses": -2.5,
    "guidance raised": 2.2,
    "raises guidance": 2.2,
    "cuts guidance": -2.2,
    "guidance cut": -2.2,
    "profit warning": -2.5,
    "better than expected": 2.0,
    "worse than expected": -2.0,
    "short squeeze": 1.0,
}
NEGATORS: frozenset[str] = frozenset(
    {"not", "no", "never", "without", "isn't", "wasn't", "didn't", "doesn't", "don't",
     "fails", "failed", "fail", "unable", "hardly", "neither", "nor", "lacks"}
)  # fmt: skip
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "sharply": 1.6, "strongly": 1.5, "significantly": 1.4, "massive": 1.6,
    "huge": 1.5, "deeply": 1.5, "extremely": 1.7, "record-breaking": 1.6, "steep": 1.5,
    "slightly": 0.5, "modestly": 0.6, "marginally": 0.5, "somewhat": 0.6, "mildly": 0.6,
}  # fmt: skip

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization that keeps apostrophes and hyphens."""
    return _TOKEN_RE.findall(text.lower())


class LexiconScorer:
    """Rule-based finance sentiment scorer (no network, deterministic).

    Args:
        negation_window: How many following tokens a negator affects.
        alpha: Normalization constant; larger means scores saturate more slowly.
        lexicon: Optional override for the combined word-weight dict.
    """

    name = "lexicon"

    def __init__(
        self,
        negation_window: int = 3,
        alpha: float = 6.0,
        lexicon: Mapping[str, float] | None = None,
    ) -> None:
        self.negation_window = negation_window
        self.alpha = alpha
        self.lexicon: dict[str, float] = (
            dict(lexicon) if lexicon is not None else {**POSITIVE, **NEGATIVE}
        )
        # Index phrases by token tuple, longest first, for greedy matching.
        self._phrases = sorted(
            ((tuple(tokenize(p)), w) for p, w in PHRASES.items()),
            key=lambda pw: -len(pw[0]),
        )

    def raw_score(self, text: str) -> tuple[float, list[str]]:
        """Return the un-normalized score and the list of matched terms."""
        tokens = tokenize(text)
        total = 0.0
        hits: list[str] = []
        negate_left = 0
        scale = 1.0
        last_value = 0.0  # value of a sentiment term that ended at the previous token
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            matched: tuple[tuple[str, ...], float] | None = None
            for phrase, weight in self._phrases:
                if tuple(tokens[i : i + len(phrase)]) == phrase:
                    matched = (phrase, weight)
                    break
            if matched is not None:
                weight, step, term = matched[1], len(matched[0]), " ".join(matched[0])
            elif tok in self.lexicon:
                weight, step, term = self.lexicon[tok], 1, tok
            else:
                weight, step, term = 0.0, 1, ""

            if weight:
                value = weight * scale
                if negate_left > 0:
                    value *= -0.75  # negated positives are less negative than true negatives
                    term = f"not {term}"
                total += value
                hits.append(term)
                scale = 1.0
                last_value = value
                negate_left = max(0, negate_left - step)
                i += step
                continue
            if tok in NEGATORS:
                negate_left = self.negation_window + 1
            elif tok in INTENSIFIERS:
                factor = INTENSIFIERS[tok]
                if last_value:  # post-modifier, e.g. "shares fall sharply"
                    total += last_value * (factor - 1.0)
                else:  # pre-modifier, e.g. "sharply lower"
                    scale = factor
            last_value = 0.0
            negate_left = max(0, negate_left - step)
            i += step
        return total, hits

    def score_text(self, text: str) -> SentimentResult:
        """Score a single text into [-1, 1]."""
        raw, hits = self.raw_score(text)
        score = raw / math.sqrt(raw * raw + self.alpha) if raw else 0.0
        reason = "matched: " + ", ".join(hits) if hits else "no lexicon terms matched"
        return SentimentResult(score=score, label=label_for(score), reason=reason)

    def score_texts(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Score many texts; see :meth:`score_text`."""
        return [self.score_text(t) for t in texts]
