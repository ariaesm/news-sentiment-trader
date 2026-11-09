PYTHON ?= python3
RUFF ?= ruff
export PYTHONPATH := src

.PHONY: help install test lint format data demo docs clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint and check formatting
	$(RUFF) check .
	$(RUFF) format --check .

format:  ## Auto-format and fix lint
	$(RUFF) format .
	$(RUFF) check --fix .

data:  ## Regenerate the synthetic sample data
	$(PYTHON) scripts/make_sample_data.py --seed 42

demo:  ## Run the full pipeline on the synthetic data (lexicon scorer)
	$(PYTHON) -m news_sentiment_trader.cli run-all --news data/sample_news.json \
		--prices data/sample_prices.csv --scorer lexicon --out reports

docs: demo  ## Refresh the README example charts in docs/
	cp reports/*.png docs/

clean:  ## Remove caches and generated outputs
	rm -rf reports outputs .cache .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
