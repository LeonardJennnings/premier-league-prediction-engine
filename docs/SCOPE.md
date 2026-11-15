# Scope

## Goal

Build a small, honest Premier League match-outcome forecasting stack: download historical results and odds, clean them, fit simple models, evaluate on a fixed test set against bookmaker implied probabilities, and show the results in a Streamlit dashboard that can also run in Docker.

## In scope

- Historical Premier League CSVs from football-data.co.uk (configured seasons)
- Clean schema, implied odds, and basic features
- Poisson goals model and a scikit-learn logistic baseline
- Fixed backtest (train before season `2526`, test on `2526`)
- Metrics: log loss, Brier, accuracy; comparison to naive frequencies and bookmakers
- Streamlit dashboard for match probabilities and eval charts
- Logging, `.env` template for secrets, Dockerfile + volume-mounted data
- Tests with pytest; README, this scope note, and a short case study

## Out of scope

- Live in-play odds or tipster / gambling advice
- Claiming to beat the market as a product
- Every league worldwide, player-level injury models, or full Dixon–Coles as a requirement
- Kubernetes, multi-service DevOps, or paid cloud infrastructure
- Redistributing football-data.co.uk raw files via Git

## Assumptions

- Closing Bet365-style odds columns are available on test matches used for bookmaker comparison
- A laptop with Python 3.11+ (or Docker Desktop) is enough to reproduce the demo
- Processed data is produced locally via the pipeline; raw CSVs stay gitignored

## Done means

- `pytest` passes
- `results/eval_results.csv` exists and is explained in the README
- Dashboard runs locally (`streamlit run`) and via Docker with mounted `data/`
- Scope and case study docs exist; repo is pushed to GitHub
