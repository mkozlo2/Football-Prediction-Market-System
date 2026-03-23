from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def assess_pipeline_health(
    matches: pd.DataFrame,
    model_metrics: dict[str, float],
    config: dict,
) -> dict[str, object]:
    max_data_age_days = int(config.get("max_data_age_days", 14))
    min_accuracy = float(config.get("min_accuracy", 0.45))
    max_log_loss = float(config.get("max_log_loss", 1.10))

    issues: list[str] = []
    last_match_date = None

    if not matches.empty and "Date" in matches.columns:
        last_match_date = pd.to_datetime(matches["Date"]).max()
        if pd.notna(last_match_date):
            current_utc = pd.Timestamp.utcnow().tz_localize(None)
            age_days = (current_utc - last_match_date).days
            if age_days > max_data_age_days:
                issues.append(
                    f"data_stale:last_match_age_days={age_days}>allowed={max_data_age_days}"
                )

    accuracy = float(model_metrics.get("accuracy", 0.0))
    log_loss = float(model_metrics.get("log_loss", 0.0))
    if accuracy < min_accuracy:
        issues.append(f"accuracy_below_threshold:{accuracy:.4f}<{min_accuracy:.4f}")
    if log_loss > max_log_loss:
        issues.append(f"log_loss_above_threshold:{log_loss:.4f}>{max_log_loss:.4f}")

    return {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok" if not issues else "warning",
        "issues": issues,
        "last_match_date": None if last_match_date is None else str(last_match_date),
        "accuracy": accuracy,
        "log_loss": log_loss,
    }


def write_health_log(health_summary: dict[str, object], log_dir: str | Path) -> Path:
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline_health.json"
    log_path.write_text(json.dumps(health_summary, indent=2), encoding="utf-8")
    return log_path
