# Premier League Prediction Engine

A learning project that downloads Premier League results, cleans them, and builds a simple match prediction pipeline with a Streamlit dashboard.

Run the pipeline later with: `python3 -m src.pipeline`

## Setup

1. Create the virtual environment:

```bash
python3 -m venv .venv
```

2. Activate it (macOS/Linux):

```bash
source .venv/bin/activate 
```

3. Activate it (Windows PowerShell):

```bash
.venv\Scripts\Activate.ps1 
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt 
```

## Data

Match results and odds are from [football-data.co.uk](https://www.football-data.co.uk/).
Column meanings: [notes.txt](https://www.football-data.co.uk/notes.txt).
Raw CSVs are downloaded into `data/raw/` and are not committed to Git.

## Tests

With the venv active, from the project root:

```bash
pytest
```