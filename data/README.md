# Sample data (SYNTHETIC)

**Everything in this folder is synthetic.** None of it is real market data or real news.

| File | Contents |
|------|----------|
| `sample_prices.csv` | 250 business days (from 2025-01-02) of simulated daily closes for `BTC-USD`, `AAPL`, `MSFT` in long format (`date,ticker,close`). Random walks — they do not track the real assets. |
| `sample_news.json` | Template-generated headlines. The tone of each day's headlines is drawn from a latent shock that is weakly correlated (ρ = 0.3 before headline noise) with the *next* day's return, so the pipeline has a known, planted effect to recover. |

Regenerate deterministically with:

```bash
python scripts/make_sample_data.py --seed 42
```

Ticker names are used only so that the keyword tagger has something familiar to match.
Results computed on this data are illustrative only and say nothing about real markets.
