from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_matches, load_matches_from_db, persist_matches


def test_load_matches_raises_for_missing_core_columns(tmp_path: Path) -> None:
    broken_csv = tmp_path / "2324" / "E0.csv"
    broken_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Date": "01/01/2024", "HomeTeam": "Alpha"}]).to_csv(broken_csv, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_matches([broken_csv])


def test_load_matches_and_persist_round_trip(tmp_path: Path) -> None:
    valid_csv = tmp_path / "2324" / "E0.csv"
    valid_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Date": "01/01/2024",
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTR": "H",
                "FTHG": 2,
                "FTAG": 1,
                "HS": 10,
                "AS": 8,
                "HST": 5,
                "AST": 3,
            }
        ]
    ).to_csv(valid_csv, index=False)

    matches = load_matches([valid_csv])
    db_path = tmp_path / "pipeline.db"
    persist_matches(matches, db_path)
    reloaded = load_matches_from_db(db_path)

    assert len(matches) == 1
    assert reloaded.loc[0, "HomeTeam"] == "Alpha"
    assert pd.to_datetime(reloaded.loc[0, "Date"]) == pd.Timestamp("2024-01-01")
