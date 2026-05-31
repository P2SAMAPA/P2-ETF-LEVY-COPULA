# Lévy Copula – Tail Dependence Engine for ETFs

Implements empirical tail dependence (Lévy copula) to measure the probability of joint extreme negative moves between ETFs. The per‑ETF score is the average tail dependence with all other ETFs – a measure of systemic risk or diversification potential.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Lower tail dependence defined as P(rank <= p | other rank <= p) with p=0.05
- Score = average pairwise tail dependence (excluding self)
- High score → ETF tends to crash together with others (systemic)
- Low score → ETF crashes independently (diversifier)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-levy-copula-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The engine provides a complementary signal to your univariate `LEVY-STABLE` engine.
- Use high‑tail‑dependence ETFs for hedging (they move together) or low‑tail‑dependence for diversification.

## Requirements

See `requirements.txt`.
