# Premier League Prediction Engine

Historical Premier League results and bookmaker odds are messy to turn into a fair, repeatable forecast demo. This project downloads those CSVs, cleans them, fits simple match-outcome models, evaluates them on a fixed test set against the market, and shows the results in Streamlit.

This is a **forecast / engineering demo**, not a betting tipster or gambling product.

![Dashboard screenshot](docs/dashboard.png)

## Architecture

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

More detail: [docs/CASE_STUDY.md](docs/CASE_STUDY.md) · [docs/SCOPE.md](docs/SCOPE.md)

## Project layout

```text
app/           Streamlit dashboard
src/           Library code (download, clean, features, models, evaluate, pipeline)
tests/         pytest suite
data/raw/      Downloaded CSVs (gitignored)
data/processed/  Cleaned matches (gitignored)
results/       Eval tables (committed)
docs/          Scope, case study, dashboard screenshot
SYLLABUS.md    Teaching syllabus + future-scope epics
```

## Setup (macOS / Linux / Windows)

1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/LeonardJennnings/premier-league-prediction-engine.git
cd premier-league-prediction-engine
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

4. (Optional) copy the env template — never commit `.env`:

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

## Results

Fixed backtest: train on seasons `2223`–`2425` (1,140 matches), test on `2526` (380 matches with closing odds). Lower log loss / Brier is better.

| Model | Log loss | Brier | Accuracy |
|-------|----------|-------|----------|
| Naive frequencies | 1.085 | 0.219 | 42.6% |
| Poisson | 1.041 | 0.209 | 45.0% |
| Sklearn baseline | 1.016 | 0.204 | 49.5% |
| Bookmaker implied | 1.013 | 0.203 | 49.7% |

**Reading:** both models beat a naive “always use historical H/D/A rates” baseline. The sklearn baseline lands very close to the bookmaker row; the Poisson goals model improves on naive but sits a little behind the market. That is an honest engineering result — matching closing odds is hard, and a dramatic unexplained win would be more suspicious than useful. Full numbers live in `results/eval_results.csv`.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Run from the **project root**. Open the local URL Streamlit prints (usually http://localhost:8501).

## Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). Build once from the project root:

```bash
docker build -t pl-engine .
```

**Data:** the image does not include `data/raw/` (and should not redistribute those files). Run the local pipeline first so `data/processed/matches.csv` exists, then **mount** your laptop’s `data/` (and optionally `results/`) into the container:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  pl-engine
```

Open http://localhost:8501. `-p 8501:8501` publishes the Streamlit port; `-v` means the container reads/writes your local folders instead of baking stale data into the image. Stop with Ctrl+C.

## Tests

```bash
pytest
```

## Demo

Record a ~2 minute screen walkthrough (README → app → one match → eval table) and host it unlisted on Loom or YouTube. Add the link here when you have it:

**Demo video:** _add Loom/YouTube URL_

## Optional deploy

[Streamlit Community Cloud](https://streamlit.io/cloud) can host `app/streamlit_app.py` from this GitHub repo. Put secrets in the Cloud secrets UI, not in Git. The deployed app still needs processed data available somehow (committed sample, build step, or documented download) — see the case study if you extend this.

## Data

Match results and odds are from [football-data.co.uk](https://www.football-data.co.uk/).
Column meanings: [notes.txt](https://www.football-data.co.uk/notes.txt).
Raw CSVs live in `data/raw/` and are not committed to Git. This project does **not** claim a licence over that source data.

## Licence

Project code is under the [MIT License](LICENSE). football-data.co.uk data remains subject to that site’s terms; do not treat the CSVs as yours to redistribute.
