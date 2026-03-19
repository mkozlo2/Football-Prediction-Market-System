# Football Prediction Market Trading System

An end-to-end Python project for building a **football match outcome model**, comparing it to bookmaker pricing, generating trading signals, sizing bets with **fractional Kelly**, and running a backtest with a bankroll curve.

## 1) Architecture

```text
                    ┌──────────────────────────────┐
                    │ football-data.co.uk CSVs     │
                    │ historical matches + odds    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ data_loader.py               │
                    │ download + clean raw data    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ features.py                  │
                    │ rolling form + Elo + stats   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ modeling.py                  │
                    │ calibrated multinomial model │
                    └──────────────┬───────────────┘
                                   │ model probs
                                   ▼
                    ┌──────────────────────────────┐
                    │ strategy.py                  │
                    │ market probs vs model probs  │
                    │ edge detection + Kelly size  │
                    └──────────────┬───────────────┘
                                   │ trade signals
                                   ▼
                    ┌──────────────────────────────┐
                    │ backtest.py                  │
                    │ PnL + bankroll simulation    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ reporting.py                 │
                    │ CSV logs + JSON + charts     │
                    └──────────────────────────────┘
```

## 2) Project structure

```text
football_prediction_market_system/
├── config.yaml
├── main.py
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── backtest.py
    ├── data_loader.py
    ├── features.py
    ├── modeling.py
    ├── reporting.py
    ├── settings.py
    └── strategy.py
```

## 3) What the system does

- Downloads historical football results and odds
- Builds pre-match features only from past information
- Trains a calibrated 3-class model for:
  - Home win
  - Draw
  - Away win
- Converts bookmaker odds into implied market probabilities
- Identifies positive-edge bets where:
  - `model probability > market probability + threshold`
- Sizes positions using **fractional Kelly**
- Runs a bankroll backtest and saves reports

## 4) Step-by-step setup

### Step 1: Install Python
Use **Python 3.11+**.

Check version:

```bash
python --version
```

### Step 2: Create a project folder
Either unzip this project or clone/copy it into a folder.

### Step 3: Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the full pipeline

```bash
python main.py --config config.yaml
```

This will:
- download match CSVs into `data/raw/`
- train the model
- run the strategy backtest
- save charts and logs into `artifacts/`

## 5) Output files

After running, check:

```text
artifacts/models/
  outcome_model.joblib
  feature_columns.json
  metrics.json

artifacts/reports/
  trade_log.csv
  model_metrics.json
  backtest_summary.json
  equity_curve.png
  edge_distribution.png
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
