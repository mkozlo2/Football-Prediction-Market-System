# Football Prediction Market Trading System

An end-to-end machine learning and trading simulation project that predicts football match outcomes, converts those predictions into probabilities, compares them against bookmaker odds, and executes simulated bets with bankroll tracking.

This project is designed to demonstrate more than just prediction accuracy. It shows how to turn model outputs into decisions under uncertainty, which makes it a stronger portfolio piece for data, ML, analytics, and trading-adjacent roles.

## Why this project stands out

- builds a full pipeline rather than only a model
- uses leakage-safe pre-match feature engineering
- produces calibrated probabilities, not just class labels
- compares model prices against market-implied prices
- sizes bets with fractional Kelly
- evaluates strategy performance through PnL and bankroll backtesting

## What it does

The system:

1. loads historical football match and bookmaker data
2. engineers rolling form and Elo-based features using only past information
3. trains a calibrated multiclass outcome model for home / draw / away
4. converts bookmaker odds into implied probabilities
5. identifies value bets where model probability exceeds market probability
6. simulates trades and tracks bankroll over time
7. exports reports, metrics, and charts

## Recruiter summary

Built an end-to-end football prediction market trading system in Python that ingests historical match and bookmaker data, engineers leakage-safe rolling and Elo features, trains a calibrated probabilistic classifier, compares model probabilities against market-implied prices, sizes trades with fractional Kelly, and evaluates performance with bankroll backtesting and reporting.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- PyYAML
- pytest
- Ruff
- Docker

## Pipeline overview

```text
Historical football data + odds
        -> data loading and cleaning
        -> rolling form + Elo features
        -> calibrated outcome model
        -> market probability comparison
        -> edge detection + Kelly sizing
        -> backtest and reporting
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

## Data source

Historical match and odds data comes from [football-data.co.uk](https://www.football-data.co.uk/).

The default configuration includes:

- Premier League: `E0`
- Bundesliga: `D1`
- La Liga: `SP1`
- Serie A: `I1`
- Ligue 1: `F1`

Across these seasons:

- `2022/23`
- `2023/24`
- `2024/25`

## Features and modeling

The model uses only information available before kickoff, including:

- rolling goals for and against
- rolling shots and shots on target
- rolling points form
- home-away feature differences
- Elo ratings and Elo differentials

The modeling pipeline includes:

- median imputation
- feature scaling
- multinomial logistic regression
- probability calibration with `CalibratedClassifierCV`

## Strategy logic

For each match:

1. predict home, draw, and away probabilities
2. infer market probabilities from bookmaker odds
3. remove overround
4. calculate edge between model and market probabilities
5. place a simulated bet only when edge exceeds a threshold
6. size the stake with fractional Kelly

## Outputs

Running the pipeline generates artifacts such as:

- trained model files
- saved feature column metadata
- model metrics
- trade log CSV
- backtest summary JSON
- bankroll curve chart
- edge distribution chart

## Quick start

### Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --config config.yaml
```

Generated files are written to:

- `artifacts/models/`
- `artifacts/reports/`

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

## Testing and linting

Install development dependencies:

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


