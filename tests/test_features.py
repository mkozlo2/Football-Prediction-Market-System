from __future__ import annotations

import pandas as pd

from src.features import build_features


def test_build_features_uses_only_past_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "0",
                "Date": pd.Timestamp("2024-01-01"),
                "league_code": "E0",
                "season_code": "2324",
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTR": "H",
                "FTHG": 2,
                "FTAG": 1,
                "HS": 10,
                "AS": 8,
                "HST": 5,
                "AST": 3,
                "B365H": 2.0,
                "B365D": 3.2,
                "B365A": 4.0,
            },
            {
                "match_id": "1",
                "Date": pd.Timestamp("2024-01-08"),
                "league_code": "E0",
                "season_code": "2324",
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTR": "D",
                "FTHG": 0,
                "FTAG": 0,
                "HS": 7,
                "AS": 6,
                "HST": 2,
                "AST": 2,
                "B365H": 2.1,
                "B365D": 3.1,
                "B365A": 3.9,
            },
        ]
    )

    feature_df = build_features(matches, rolling_windows=[3, 5], use_elo=False)

    first_match = feature_df.iloc[0]
    second_match = feature_df.iloc[1]

    assert first_match["home_avg_gf_3"] == 0.0
    assert first_match["away_avg_ga_3"] == 0.0
    assert second_match["home_avg_gf_3"] == 2.0
    assert second_match["away_avg_ga_3"] == 2.0
    assert second_match["home_avg_points_3"] == 3.0
    assert second_match["away_avg_points_3"] == 0.0


def test_build_features_creates_target_column() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "0",
                "Date": pd.Timestamp("2024-02-01"),
                "league_code": "E0",
                "season_code": "2324",
                "HomeTeam": "Alpha",
                "AwayTeam": "Gamma",
                "FTR": "A",
                "FTHG": 1,
                "FTAG": 2,
                "HS": 9,
                "AS": 11,
                "HST": 3,
                "AST": 4,
            }
        ]
    )

    feature_df = build_features(matches, rolling_windows=[5], use_elo=True)

    assert feature_df.loc[0, "target"] == 2
    assert "elo_diff" in feature_df.columns
