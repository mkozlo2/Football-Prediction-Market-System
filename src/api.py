from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling import SavedModelBundle, load_saved_model
from src.strategy import kelly_fraction, remove_overround


DEFAULT_MODEL_DIR = Path("artifacts/models")
OUTCOME_KEYS = ("home", "draw", "away")


class OddsPayload(BaseModel):
    home: float = Field(..., gt=1.0)
    draw: float = Field(..., gt=1.0)
    away: float = Field(..., gt=1.0)


class PredictRequest(BaseModel):
    features: dict[str, float | None]
    odds: OddsPayload
    bankroll: float = Field(1000.0, gt=0.0)
    edge_threshold: float = Field(0.03, ge=0.0)
    max_fractional_kelly: float = Field(0.25, gt=0.0)
    max_bet_fraction_of_bankroll: float = Field(0.05, gt=0.0)
    min_decimal_odds: float = Field(1.5, gt=1.0)
    max_decimal_odds: float = Field(6.0, gt=1.0)


def build_prediction_frame(
    feature_values: dict[str, float | None],
    feature_columns: list[str],
) -> pd.DataFrame:
    row = {column: feature_values.get(column, np.nan) for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def price_match(
    request: PredictRequest,
    bundle: SavedModelBundle,
) -> dict[str, Any]:
    frame = build_prediction_frame(request.features, bundle.feature_columns)
    probabilities = bundle.model.predict_proba(frame)[0]

    market_odds = np.array(
        [request.odds.home, request.odds.draw, request.odds.away],
        dtype=float,
    )
    market_probs = remove_overround(1.0 / market_odds)
    edges = probabilities - market_probs

    best_idx = int(np.argmax(edges))
    best_edge = float(edges[best_idx])
    chosen_odds = float(market_odds[best_idx])
    raw_kelly = 0.0
    stake_fraction = 0.0
    stake_amount = 0.0
    action = "HOLD"

    if (
        best_edge >= request.edge_threshold
        and request.min_decimal_odds <= chosen_odds <= request.max_decimal_odds
    ):
        raw_kelly = kelly_fraction(float(probabilities[best_idx]), chosen_odds)
        stake_fraction = min(
            raw_kelly * request.max_fractional_kelly,
            request.max_bet_fraction_of_bankroll,
        )
        stake_amount = request.bankroll * stake_fraction
        if stake_fraction > 0:
            action = f"BET_{OUTCOME_KEYS[best_idx].upper()}"

    return {
        "model_probabilities": {
            OUTCOME_KEYS[idx]: float(probabilities[idx]) for idx in range(3)
        },
        "market_probabilities": {
            OUTCOME_KEYS[idx]: float(market_probs[idx]) for idx in range(3)
        },
        "edges": {OUTCOME_KEYS[idx]: float(edges[idx]) for idx in range(3)},
        "recommendation": {
            "action": action,
            "outcome": OUTCOME_KEYS[best_idx] if action != "HOLD" else "none",
            "best_edge": best_edge,
            "chosen_odds": chosen_odds,
            "kelly_fraction_raw": raw_kelly,
            "stake_fraction": stake_fraction,
            "stake_amount": stake_amount,
        },
    }


@lru_cache(maxsize=1)
def get_model_bundle(model_dir: str = str(DEFAULT_MODEL_DIR)) -> SavedModelBundle:
    return load_saved_model(model_dir)


app = FastAPI(
    title="Football Prediction Market API",
    version="1.0.0",
    description="Desk-facing API for pricing football matches and generating trade signals.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Saved model artifacts were not found. "
                "Run the training pipeline before starting the API."
            ),
        ) from exc

    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(bundle.feature_columns),
        "metrics": bundle.metrics,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Saved model artifacts were not found. "
                "Run the training pipeline before calling /predict."
            ),
        ) from exc

    return price_match(request, bundle)
