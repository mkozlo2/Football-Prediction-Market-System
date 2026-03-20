from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_EXCLUDE = {
    "match_id",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "FTHG",
    "FTAG",
    "target",
    "league_code",
    "season_code",
}


@dataclass
class ModelArtifacts:
    model: CalibratedClassifierCV
    feature_columns: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    test_proba: np.ndarray
    metrics: dict[str, float]


def train_outcome_model(feature_df: pd.DataFrame, config: dict) -> ModelArtifacts:
    df = feature_df.sort_values("Date").reset_index(drop=True).copy()
    df = df.dropna(subset=["target"])

    feature_columns = [
        c for c in df.columns
        if c not in FEATURE_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns available after exclusions.")

    split_mode = config.get("split_mode", "time")
    test_size = float(config.get("test_size", 0.2))

    if split_mode == "season" and "season_code" in df.columns and df["season_code"].notna().any():
        latest_season = df["season_code"].dropna().iloc[-1]
        train_df = df[df["season_code"] != latest_season].copy()
        test_df = df[df["season_code"] == latest_season].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Season-based split failed because train or test set is empty. "
                "Check season_code values or fall back to split_mode='time'."
            )
    else:
        split_idx = int(len(df) * (1.0 - test_size))
        if split_idx <= 0 or split_idx >= len(df):
            raise ValueError("Invalid split index. Check test_size and dataset size.")
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=config.get("max_iter", 500),
            random_state=config.get("random_state", 42),
        )),
    ])

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method=config.get("calibration_method", "sigmoid"),
        cv=config.get("calibration_cv", 3),
    )

    calibrated.fit(X_train, y_train)
    test_proba = calibrated.predict_proba(X_test)
    test_pred = np.argmax(test_proba, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_proba)),
        "brier_home": float(brier_score_loss((y_test == 0).astype(int), test_proba[:, 0])),
        "brier_draw": float(brier_score_loss((y_test == 1).astype(int), test_proba[:, 1])),
        "brier_away": float(brier_score_loss((y_test == 2).astype(int), test_proba[:, 2])),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_features": int(len(feature_columns)),
    }

    return ModelArtifacts(
        model=calibrated,
        feature_columns=feature_columns,
        train_df=train_df,
        test_df=test_df,
        test_proba=test_proba,
        metrics=metrics,
    )


def save_model(artifacts: ModelArtifacts, model_dir: str | Path) -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, model_path / "outcome_model.joblib")
    (model_path / "feature_columns.json").write_text(
        json.dumps(artifacts.feature_columns, indent=2),
        encoding="utf-8",
    )
    (model_path / "metrics.json").write_text(
        json.dumps(artifacts.metrics, indent=2),
        encoding="utf-8",
    )
