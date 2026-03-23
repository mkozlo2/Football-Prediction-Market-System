from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.storage import read_table, write_table

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

ODDS_CANDIDATES = [
    "B365H",
    "B365D",
    "B365A",
    "AvgH",
    "AvgD",
    "AvgA",
]


def _download_csv(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def ensure_data(
    leagues: Iterable[str],
    seasons: Iterable[str],
    cache_dir: str | Path,
) -> list[Path]:
    cache_path = Path(cache_dir)
    files: list[Path] = []
    for season in seasons:
        for league in leagues:
            out_file = cache_path / season / f"{league}.csv"
            if not out_file.exists():
                url = BASE_URL.format(season=season, league=league)
                print(f"Downloading {url}")
                _download_csv(url, out_file)
            files.append(out_file)
    return files


def load_matches(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        keep_cols = [c for c in REQUIRED_COLUMNS + ODDS_CANDIDATES if c in df.columns]
        missing_core = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTR"] if c not in df.columns]
        if missing_core:
            raise ValueError(f"Missing required columns {missing_core} in {path}")
        trimmed = df[keep_cols].copy()
        trimmed = trimmed.assign(
            source_file=path.name,
            league_code=path.stem,
            season_code=path.parent.name,
        )
        frames.append(trimmed)

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["Date"] = pd.to_datetime(all_matches["Date"], dayfirst=True, errors="coerce")
    all_matches = all_matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])

    numeric_cols = [
        c
        for c in REQUIRED_COLUMNS + ODDS_CANDIDATES
        if c in all_matches.columns and c not in {"Date", "HomeTeam", "AwayTeam", "FTR"}
    ]
    for col in numeric_cols:
        all_matches[col] = pd.to_numeric(all_matches[col], errors="coerce")

    all_matches = all_matches.sort_values("Date").reset_index(drop=True)
    all_matches["match_id"] = all_matches.index.astype(str)
    return all_matches


def persist_matches(matches: pd.DataFrame, db_path: str | Path) -> None:
    write_table(matches, db_path, "matches")


def load_matches_from_db(db_path: str | Path) -> pd.DataFrame:
    matches = read_table(db_path, "SELECT * FROM matches ORDER BY Date")
    if "Date" in matches.columns:
        matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    return matches
