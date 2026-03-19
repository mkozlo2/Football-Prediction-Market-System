from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd


def save_reports(results: pd.DataFrame, model_metrics: dict[str, float], backtest_summary: dict[str, float], report_dir: str | Path) -> None:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "trade_log.csv", index=False)
    (out_dir / "model_metrics.json").write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")
    (out_dir / "backtest_summary.json").write_text(json.dumps(backtest_summary, indent=2), encoding="utf-8")

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
