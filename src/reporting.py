from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.storage import write_json_record, write_table


def save_reports(
    feature_df: pd.DataFrame,
    signals: pd.DataFrame,
    results: pd.DataFrame,
    model_metrics: dict[str, float],
    backtest_summary: dict[str, float],
    report_dir: str | Path,
    db_path: str | Path,
    health_summary: dict[str, object],
) -> None:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(out_dir / "features_snapshot.csv", index=False)
    signals.to_csv(out_dir / "signals.csv", index=False)
    results.to_csv(out_dir / "trade_log.csv", index=False)
    (out_dir / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2),
        encoding="utf-8",
    )
    (out_dir / "backtest_summary.json").write_text(
        json.dumps(backtest_summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "health_summary.json").write_text(
        json.dumps(health_summary, indent=2),
        encoding="utf-8",
    )

    write_table(feature_df, db_path, "features_snapshot")
    write_table(signals, db_path, "signals")
    write_table(results, db_path, "backtest_results")
    write_json_record(db_path, "model_metrics", model_metrics)
    write_json_record(db_path, "backtest_summary", backtest_summary)
    write_json_record(db_path, "health_summary", health_summary)

    if not results.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(results["Date"]), results["bankroll_after"])
        plt.title("Bankroll Over Time")
        plt.xlabel("Date")
        plt.ylabel("Bankroll")
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=150)
        plt.close()

        placed = results[results["signal"] != "HOLD"].copy()
        if not placed.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(placed["best_edge"], bins=20)
            plt.title("Distribution of Trading Edges")
            plt.xlabel("Edge")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / "edge_distribution.png", dpi=150)
            plt.close()
