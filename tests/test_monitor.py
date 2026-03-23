from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.monitor import assess_pipeline_health, write_health_log


def test_assess_pipeline_health_flags_stale_or_degraded_pipeline() -> None:
    matches = pd.DataFrame([{"Date": pd.Timestamp("2023-01-01")}])
    metrics = {"accuracy": 0.30, "log_loss": 1.5}
    config = {"max_data_age_days": 7, "min_accuracy": 0.45, "max_log_loss": 1.1}

    summary = assess_pipeline_health(matches, metrics, config)

    assert summary["status"] == "warning"
    assert len(summary["issues"]) >= 2


def test_write_health_log_creates_json_file(tmp_path: Path) -> None:
    summary = {"status": "ok", "issues": []}

    log_path = write_health_log(summary, tmp_path)

    assert log_path.exists()
    assert '"status": "ok"' in log_path.read_text(encoding="utf-8")
