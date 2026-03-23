from __future__ import annotations

import numpy as np

from src.api import PredictRequest, build_prediction_frame, price_match
from src.modeling import SavedModelBundle


class DummyModel:
    def predict_proba(self, frame):  # noqa: ANN001
        assert list(frame.columns) == ["feature_a", "feature_b"]
        return np.array([[0.55, 0.20, 0.25]])


def test_build_prediction_frame_aligns_to_feature_columns() -> None:
    frame = build_prediction_frame(
        {"feature_b": 2.0},
        ["feature_a", "feature_b"],
    )

    assert list(frame.columns) == ["feature_a", "feature_b"]
    assert np.isnan(frame.loc[0, "feature_a"])
    assert frame.loc[0, "feature_b"] == 2.0


def test_price_match_returns_bet_recommendation() -> None:
    request = PredictRequest(
        features={"feature_a": 1.0},
        odds={"home": 2.5, "draw": 3.2, "away": 3.0},
        bankroll=1000.0,
    )
    bundle = SavedModelBundle(
        model=DummyModel(),
        feature_columns=["feature_a", "feature_b"],
        metrics={"accuracy": 0.5},
    )

    response = price_match(request, bundle)

    assert response["recommendation"]["action"] == "BET_HOME"
    assert response["recommendation"]["stake_amount"] > 0
    assert response["model_probabilities"]["home"] == 0.55
