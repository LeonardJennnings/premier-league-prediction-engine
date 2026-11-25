# Case study — Premier League Prediction Engine

## Problem

Analysts and curious fans often want match probabilities that are transparent and comparable to the market, not opaque tips. Rebuilding downloads, cleaning, and charts by hand each week is slow and easy to get wrong.

## What we built

A Python project that:

1. Downloads (or skips) season CSVs into `data/raw/`
2. Cleans them into `data/processed/matches.csv`
3. Fits a Poisson goals model and a scikit-learn baseline
4. Scores both on a fixed test season against naive frequencies and bookmaker implied odds
5. Serves the same logic through a Streamlit app and an optional Docker image

Architecture (same flow as the teaching syllabus):

```mermaid
flowchart LR
    subgraph config [config.py]
        Settings["Paths, season codes, URLs"]
    end

    subgraph ingest [Ingest]
        Download["download.py"]
        Clean["clean.py"]
        Features["features.py"]
    end

    subgraph model [Models]
        Poisson["models/poisson.py"]
        Baseline["models/baseline.py"]
    end

    subgraph out [Outputs]
        Evaluate["evaluate.py"]
        Results["results/ tables"]
        App["app/streamlit_app.py"]
    end

    Pipeline["pipeline.py"] --> Download
    Settings --> Download
    Settings --> Clean
    Settings --> Evaluate
    Download --> Clean
    Clean --> Features
    Features --> Poisson
    Features --> Baseline
    Poisson --> Evaluate
    Baseline --> Evaluate
    Evaluate --> Results
    Features --> App
    Poisson --> App
    Baseline --> App
```

## Data and models

- **Data:** football-data.co.uk Premier League (`E0`) files; raw files not committed.
- **Poisson:** team attack/defence style goals rates → home/draw/away probabilities.
- **Baseline:** logistic regression using bookmaker-implied features (fit on train only).
- **Bookmaker row:** normalised closing odds on the same test matches — a tough external baseline, not a “target to beat at all costs.”

## Results

On season `2526` (380 test matches; 1,140 train matches from prior seasons):

| Model | Log loss | Brier | Accuracy |
|-------|----------|-------|----------|
| Naive frequencies | 1.085 | 0.219 | 42.6% |
| Poisson | 1.034 | 0.208 | 47.6% |
| Sklearn baseline | 1.016 | 0.204 | 49.5% |
| Bookmaker implied | 1.013 | 0.203 | 49.7% |

The Poisson away rate uses the average number of away goals. Sklearn sits next to the bookmaker row. Poisson is closer to the bookmaker row than the naive frequencies line is, and it is still behind that row on this one season. Limitations: one league, simple features, no injuries or line-ups, and odds-based features mean the sklearn baseline is partly learning the market. The research comparison, with a match count and a range, is the walk-forward paired table.

## ROI / time-saved framing

There is no paying client here. The concrete win is engineering time: one pipeline command regenerates processed data and sample forecasts; one eval command regenerates a comparable metrics table; the dashboard and Docker path make the demo repeatable without re-explaining a personal laptop setup. That is hours of fragile notebook work turned into a short, documented runbook.

## Will it break?

- **Tests:** pytest covers cleaning/features/models/eval helpers.
- **Eval:** fixed train/test rule in `results/EVAL_README.md` — do not retune the split to chase scores.
- **Logging:** download/pipeline/evaluate use structured logs instead of silent failure.
- **Downloads:** idempotent skip if files exist; network failures surface as errors.
- **Leakage:** models fit on train only; features must not use future match results.
- **UI language:** probabilities and forecasts, not tips or certainties.
- **Secrets:** `.env` never baked into Docker images.

**Next (future scope, not this delivery):** scheduled evals, fixtures API token usage, richer models, multi-book odds — only when measured gains justify the work.
