# Football Prediction Market Trading System

An operations-oriented trading support project that models football outcomes, compares internal prices to bookmaker markets, persists trading lifecycle data in SQL, monitors pipeline health, and produces simulated trading decisions with bankroll tracking.

This project is intentionally framed as more than an ML model. It is a compact example of designing, building, testing, and maintaining an internal trading tool that could sit near a desk workflow.

## Why this project is relevant

This repository is designed to show capability at the intersection of:

- trading logic
- Python engineering
- SQL-backed data workflows
- testing and maintainability
- operational ownership across the full lifecycle

Instead of stopping at "predict the match result," the system asks:

`How would a trading-support application ingest data, price events, compare against external counterparties, log decisions, and monitor its own health?`

## Trading operations problem

The core problem is straightforward:

- market odds imply a price for each outcome
- an internal model produces its own probability estimate
- the desk needs a repeatable way to compare those views
- any trade recommendation needs to be logged, auditable, and measurable

This system simulates that workflow by:

1. ingesting historical market and match data
2. engineering pre-match features with leakage control
3. training a calibrated model to produce internal prices
4. comparing internal prices against bookmaker prices
5. generating simulated trade signals and Kelly-sized stakes
6. storing lifecycle artifacts in SQLite for later querying
7. monitoring data freshness and model health

## Full lifecycle framing

The project is structured around ownership across the full lifecycle:

- Design: define pricing logic, edge thresholds, and persistence approach
- Build: implement a reproducible Python pipeline with SQL storage
- Test: validate feature logic, signal generation, loader behavior, and backtest edge cases
- Maintain: monitor stale data and degraded model performance

## Key design decisions

### Why calibrated probabilities?

Trading decisions depend on price quality, not just class predictions. A calibrated model is more appropriate than a raw classifier because the output needs to be interpretable as a probability before comparing it to external market prices.

### Why fractional Kelly?

Kelly sizing connects model confidence to stake sizing in a disciplined way. Using a fractional Kelly cap makes the strategy less aggressive and more realistic for risk-controlled simulation.

### Why SQLite?

The role emphasis includes SQL. This project now persists key datasets and outputs to SQLite so the workflow is queryable and closer to a real internal tool:

- raw loaded matches
- engineered feature snapshot
- trade signals
- backtest results
- summary records for model metrics and pipeline health

This keeps the setup local and simple while still demonstrating relational persistence and query-friendly design.

## Architecture

```text
Historical football data + bookmaker odds
        -> load and clean raw data
        -> persist matches to SQLite
        -> engineer rolling form and Elo features
        -> persist feature snapshot to SQLite
        -> train calibrated model
        -> compare internal probabilities vs market probabilities
        -> generate trade signals and Kelly-sized stakes
        -> backtest bankroll impact
        -> persist outputs and health summaries
        -> export reports and monitoring logs
```

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- SQLite
- matplotlib
- PyYAML
- pytest
- Ruff
- GitHub Actions
- Docker

## Project structure

```text
football_prediction_market_system/
|-- config.yaml
|-- main.py
|-- Dockerfile
|-- .dockerignore
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- src/
|   |-- api.py
|   |-- backtest.py
|   |-- data_loader.py
|   |-- features.py
|   |-- modeling.py
|   |-- monitor.py
|   |-- reporting.py
|   |-- settings.py
|   |-- storage.py
|   `-- strategy.py
|-- tests/
|   |-- test_api.py
|   |-- test_backtest.py
|   |-- test_data_loader.py
|   |-- test_features.py
|   |-- test_monitor.py
|   `-- test_strategy.py
`-- .github/workflows/ci.yml
```

## Data source

Historical match and odds data comes from [football-data.co.uk](https://www.football-data.co.uk/).

The default configuration uses:

- Premier League: `E0`
- Bundesliga: `D1`
- La Liga: `SP1`
- Serie A: `I1`
- Ligue 1: `F1`

Across these seasons:

- `2022/23`
- `2023/24`
- `2024/25`

## SQL layer

The pipeline writes key data into a local SQLite database at:

`artifacts/sql/pipeline.db`

Example stored tables:

- `matches`
- `features`
- `features_snapshot`
- `signals`
- `backtest_results`
- `model_metrics`
- `backtest_summary`
- `health_summary`

This adds a queryable persistence layer that could support downstream reconciliations, audit trails, or desk reporting.

## Monitoring and maintainability

The monitoring module checks:

- data staleness using the most recent match date
- low model accuracy relative to a configured threshold
- high log loss relative to a configured threshold

It writes a structured health log to:

`artifacts/logs/pipeline_health.json`

This is a small but useful example of how an internal ops-facing tool should surface its own health rather than silently drifting.

## Outputs

Running the pipeline produces:

- trained model artifacts
- metrics JSON files
- trade logs and signal exports
- bankroll and edge charts
- SQLite tables for lifecycle data
- structured health logs

## How this could integrate with a desk

In a fuller trading environment, the same pattern could feed downstream systems such as:

- an internal pricing or quoting dashboard
- a monitoring panel for stale data or model degradation
- an execution support tool where analysts review generated edges
- reconciliation or audit workflows backed by SQL queries

The current repository is still a local project, but the architecture is intended to point toward that operational model.

## FastAPI service

To make the project closer to a desk-facing internal application, the repository now includes a small FastAPI layer in [src/api.py](/C:/Users/mkozl/OneDrive/Dokumenty/football_prediction_market_system/src/api.py).

Available endpoints:

- `GET /health`
- `POST /predict`

The API loads the saved model artifacts from `artifacts/models/`, accepts a feature payload plus market odds, and returns:

- internal model probabilities
- market-implied probabilities
- pricing edges
- recommended action
- Kelly-based stake fraction and stake amount

Run the API locally after training the model:

```powershell
python main.py --config config.yaml
uvicorn src.api:app --reload
```

Example request:

```json
{
  "features": {
    "home_avg_gf_5": 1.8,
    "home_avg_ga_5": 0.9,
    "away_avg_gf_5": 1.1,
    "away_avg_ga_5": 1.4,
    "elo_diff": 55.0
  },
  "odds": {
    "home": 2.35,
    "draw": 3.30,
    "away": 3.10
  },
  "bankroll": 1000.0
}
```

## Quick start

### Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --config config.yaml
```

Artifacts are written to:

- `artifacts/models/`
- `artifacts/reports/`
- `artifacts/sql/`
- `artifacts/logs/`

## Docker

Build the image:

```bash
docker build -t football-prediction-market .
```

Run the pipeline in a container:

```bash
docker run --rm -v ${PWD}/artifacts:/app/artifacts football-prediction-market
```

On PowerShell, if `${PWD}` causes issues:

```powershell
docker run --rm -v "${PWD}\artifacts:/app/artifacts" football-prediction-market
```

Run the API in Docker after artifacts exist locally:

```bash
docker run --rm -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts football-prediction-market uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Testing and CI

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Run linting:

```powershell
ruff check .
```

<<<<<<< HEAD
Run tests:

```powershell
pytest
```

GitHub Actions runs both checks automatically on pushes and pull requests.

## Portfolio summary

Built an operations-oriented football prediction market system in Python that ingests historical match and bookmaker data, engineers leakage-safe features, trains a calibrated pricing model, compares internal probabilities against market-implied prices, persists lifecycle data to SQLite, monitors pipeline health, and evaluates trading decisions through Kelly-sized backtesting.

## Next upgrades

Useful next steps if I wanted to extend this further:

- expose SQL-backed analytics through a small dashboard
- add baseline comparisons and drawdown analysis
- schedule automated refresh jobs
- add alert delivery beyond JSON logs

## Notes

- This project is pre-match focused rather than in-play.
- The current strategy is a simulation for portfolio purposes, not real-money trading advice.
- Historical data quality depends on source coverage and available columns.
=======

>>>>>>> 1e48108b7e01ce475c65ff7703c84be3b2173a0d
