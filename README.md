# Premier League Prediction Engine

A learning project that downloads Premier League results, cleans them, builds Poisson and baseline match-outcome models, evaluates them on a fixed test set, and shows forecasts in a Streamlit dashboard.

This is a **forecast / engineering demo**, not betting advice.

## Project layout

```text
app/           Streamlit dashboard
src/           Library code (download, clean, features, models, evaluate, pipeline)
tests/         pytest suite
data/raw/      Downloaded CSVs (gitignored)
data/processed/  Cleaned matches (gitignored)
results/       Eval tables (committed)
docs/          Case study notes (later)
SYLLABUS.md    Teaching syllabus + future-scope epics
```

## Setup

1. Create the virtual environment:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. (Optional) copy env template and edit locally — never commit `.env`:

```bash
cp .env.example .env
```

`LOG_LEVEL` defaults to `INFO`. `FOOTBALL_DATA_ORG_TOKEN` is for a later fixtures API and can stay empty.

## Pipeline

From the project root, with the venv active:

```bash
python -m src.pipeline
```

This downloads (or skips) raw CSVs, builds `data/processed/matches.csv`, fits the models, and prints a sample comparison.

## Evaluation

```bash
python -m src.evaluate
```

Writes `results/eval_results.csv`. See `results/EVAL_README.md` for the test-set rule and metric definitions.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Run from the **project root**. Open the local URL Streamlit prints (usually http://localhost:8501).

## Tests

```bash
pytest
```

## Data

Match results and odds are from [football-data.co.uk](https://www.football-data.co.uk/).
Column meanings: [notes.txt](https://www.football-data.co.uk/notes.txt).
Raw CSVs live in `data/raw/` and are not committed to Git.
