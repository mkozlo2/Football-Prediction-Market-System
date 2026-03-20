from __future__ import annotations

import argparse

from src.backtest import run_backtest
from src.data_loader import ensure_data, load_matches
from src.features import build_features
from src.modeling import save_model, train_outcome_model
from src.reporting import save_reports
from src.settings import load_settings
from src.strategy import generate_trade_signals


def run_pipeline(config_path: str = "config.yaml") -> None:
    settings = load_settings(config_path)

    csv_paths = ensure_data(
        leagues=settings.data["leagues"],
        seasons=settings.data["seasons"],
        cache_dir=settings.data["cache_dir"],
    )
    matches = load_matches(csv_paths)
    print(f"Loaded {len(matches)} matches")

    feature_df = build_features(
        matches=matches,
        rolling_windows=settings.features["rolling_windows"],
        use_elo=settings.features.get("use_elo", True),
    )
    feature_df = feature_df.sort_values("Date").reset_index(drop=True)

    if len(feature_df) < settings.features.get("min_training_rows", 150):
        raise ValueError("Not enough rows to train. Add more seasons or leagues.")

    artifacts = train_outcome_model(feature_df, settings.model)
    save_model(artifacts, settings.outputs["model_dir"])

    signals = generate_trade_signals(artifacts.test_df, artifacts.test_proba, settings.strategy)
    results, backtest_summary = run_backtest(signals, settings.strategy["bankroll_start"])
    save_reports(results, artifacts.metrics, backtest_summary, settings.outputs["report_dir"])

    print("\nModel metrics")
    for key, value in artifacts.metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nBacktest summary")
    for key, value in backtest_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nArtifacts saved to:")
    print(f"  Models : {settings.outputs['model_dir']}")
    print(f"  Reports: {settings.outputs['report_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football Prediction Market Trading System")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
