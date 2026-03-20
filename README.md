# Football Prediction Market Trading System

An end-to-end Python project that prices football matches, compares model probabilities against bookmaker odds, generates simulated bets, and tracks bankroll performance through a backtest.

This project is designed to look and feel like a compact quant trading pipeline:

- ingest historical match and odds data
- engineer leakage-safe pre-match features
- train a calibrated multiclass outcome model
- convert bookmaker odds into implied probabilities
- place simulated trades when model edge exceeds market price
- size positions with fractional Kelly
- evaluate profit/loss, ROI, hit rate, and bankroll growth

## Why this project is useful

This is stronger than a generic sports prediction script because it connects machine learning to a decision system. Instead of stopping at accuracy, it asks the portfolio-relevant question:

`Would these predictions have led to profitable trading decisions?`

That makes it a good showcase project for roles involving:

- applied machine learning
- data pipelines
- probabilistic modeling
- decision systems
- trading or betting market analysis

## Pipeline overview

```text
Historical CSVs -> Data loading -> Feature engineering -> Calibrated model
              -> Market probability comparison -> Bet sizing -> Backtest -> Reports
```

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
|   |-- backtest.py
|   |-- data_loader.py
|   |-- features.py
|   |-- modeling.py
|   |-- reporting.py
|   |-- settings.py
|   `-- strategy.py
`-- tests/
    |-- test_backtest.py
    |-- test_features.py
    `-- test_strategy.py
```

## Data

The project uses historical football CSVs from [football-data.co.uk](https://www.football-data.co.uk/), including:

- match outcomes
- goals scored
- shots and shots on target
- bookmaker odds

By default, the pipeline is configured to use:

- Premier League: `E0`
- Bundesliga: `D1`
- La Liga: `SP1`
- Serie A: `I1`
- Ligue 1: `F1`

Across seasons:

- `2022/23`
- `2023/24`
- `2024/25`

## Features

The model builds pre-match features using only information available before kickoff:

- rolling goals for / against
- rolling shots and shots on target
- rolling points form
- home vs away feature differences
- Elo ratings and Elo difference

This keeps the modeling setup closer to a real trading environment and avoids target leakage.

## Modeling and trading logic

### Model

- multinomial logistic regression
- median imputation
- feature scaling
- probability calibration with `CalibratedClassifierCV`

### Decision rule

For each match:

1. predict home / draw / away probabilities
2. convert bookmaker odds into implied market probabilities
3. remove bookmaker overround
4. compute edge = `model_prob - market_prob`
5. bet only if the edge exceeds a threshold
6. size stake using fractional Kelly

### Backtest outputs

The backtest tracks:

- bankroll over time
- total profit
- total amount staked
- ROI
- hit rate
- trade log

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

## Docker

Build the container:

```bash
docker build -t football-prediction-market .
```

Run the pipeline:

```bash
docker run --rm -v ${PWD}/artifacts:/app/artifacts football-prediction-market
```

On PowerShell, if `${PWD}` causes issues, use:

```powershell
docker run --rm -v "${PWD}\artifacts:/app/artifacts" football-prediction-market
```

## Testing and linting

Install dev tools:

```powershell
pip install -r requirements-dev.txt
```

Run tests:

```powershell
pytest
```

Run linting:

```powershell
ruff check .
```

## 6) How to explain this in an interview

You can say:

> I built an end-to-end football prediction market trading system in Python. It ingests historical match and bookmaker data, engineers rolling form and Elo-based features, trains a calibrated probabilistic classifier, compares model probabilities against market-implied probabilities, sizes trades with fractional Kelly, and evaluates performance through a bankroll backtest and post-trade reporting.

## 7) Suggested upgrades to make this even stronger

### High-value upgrades
- Replace logistic regression with LightGBM or XGBoost
- Add team news / injuries / rest days / travel distance
- Add closing-line value analysis
- Add exchange-style commission modeling
- Build a FastAPI service for real-time pricing
- Add in-play state updates using time, score, and red cards
- Add market-making logic instead of pure directional betting

### Strong portfolio extras
- Dockerize the app
- Add unit tests
- Add CI with GitHub Actions
- Build a dashboard with Streamlit

## 8) Notes

- This MVP is **pregame-focused**, which is enough for a strong portfolio project.
- You can later extend it into **in-play trading** by adding live state features and a real-time odds feed.
- Historical data availability depends on the source columns present in the CSVs.
