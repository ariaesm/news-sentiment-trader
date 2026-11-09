# Contributing

Thanks for your interest! Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/ariaesm/news-sentiment-trader
cd news-sentiment-trader
python -m venv .venv && source .venv/bin/activate
make install
```

## Before opening a PR

```bash
make format   # ruff format + autofix
make lint     # must be clean
make test     # must pass
```

## Guidelines

- Keep runtime dependencies to numpy, pandas, scipy, matplotlib, PyYAML and the stdlib.
- Tests must not touch the network; mock the LLM transport (see `tests/test_llm_scorer.py`).
- Anything that turns news into a trading decision must respect the timing convention in
  `aggregate.py` / `backtest.py`. If you touch it, add a look-ahead test.
- Never commit real API keys; use `.env` (git-ignored) based on `.env.example`.
- Don't put results from the synthetic data in the README unless they come from actually
  running `make docs`.
