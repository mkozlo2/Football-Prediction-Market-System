from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class TeamState:
    gf: deque
    ga: deque
    shots_for: deque
    shots_against: deque
    shots_on_target_for: deque
    shots_on_target_against: deque
    points: deque
    elo: float = 1500.0


def _new_team_state(max_window: int) -> TeamState:
    return TeamState(
        gf=deque(maxlen=max_window),
        ga=deque(maxlen=max_window),
        shots_for=deque(maxlen=max_window),
        shots_against=deque(maxlen=max_window),
        shots_on_target_for=deque(maxlen=max_window),
        shots_on_target_against=deque(maxlen=max_window),
        points=deque(maxlen=max_window),
    )


def _safe_mean(items: Iterable[float]) -> float:
    values = list(items)
    return float(np.mean(values)) if values else 0.0


def _result_points(result: str, side: str) -> int:
    if result == "D":
        return 1
    if result == "H":
        return 3 if side == "home" else 0
    if result == "A":
        return 3 if side == "away" else 0
    return 0


def _expected_score(elo_a: float, elo_b: float, home_advantage: float = 60.0) -> tuple[float, float]:
    adjusted_a = elo_a + home_advantage
    exp_a = 1.0 / (1.0 + 10 ** ((elo_b - adjusted_a) / 400.0))
    exp_b = 1.0 - exp_a
    return exp_a, exp_b


def build_features(matches: pd.DataFrame, rolling_windows: list[int], use_elo: bool = True) -> pd.DataFrame:
    if not rolling_windows:
        raise ValueError("rolling_windows must contain at least one window size.")

    windows = sorted({int(window) for window in rolling_windows if int(window) > 0})
    if not windows:
        raise ValueError("rolling_windows must contain positive integers.")

    matches = matches.sort_values("Date").reset_index(drop=True).copy()
    max_window = max(windows)
    teams: dict[str, TeamState] = defaultdict(lambda: _new_team_state(max_window=max_window))
    rows: list[dict] = []

    for _, match in matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        home_state = teams[home]
        away_state = teams[away]

        row = {
            "match_id": match["match_id"],
            "Date": match["Date"],
            "league_code": match["league_code"],
            "season_code": match["season_code"],
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": match["FTR"],
            "FTHG": match.get("FTHG", np.nan),
            "FTAG": match.get("FTAG", np.nan),
        }

        for window in windows:
            row[f"home_avg_gf_{window}"] = _safe_mean(list(home_state.gf)[-window:])
            row[f"home_avg_ga_{window}"] = _safe_mean(list(home_state.ga)[-window:])
            row[f"home_avg_shots_{window}"] = _safe_mean(list(home_state.shots_for)[-window:])
            row[f"home_avg_shots_allowed_{window}"] = _safe_mean(list(home_state.shots_against)[-window:])
            row[f"home_avg_sot_{window}"] = _safe_mean(list(home_state.shots_on_target_for)[-window:])
            row[f"home_avg_sot_allowed_{window}"] = _safe_mean(list(home_state.shots_on_target_against)[-window:])
            row[f"home_avg_points_{window}"] = _safe_mean(list(home_state.points)[-window:])

            row[f"away_avg_gf_{window}"] = _safe_mean(list(away_state.gf)[-window:])
            row[f"away_avg_ga_{window}"] = _safe_mean(list(away_state.ga)[-window:])
            row[f"away_avg_shots_{window}"] = _safe_mean(list(away_state.shots_for)[-window:])
            row[f"away_avg_shots_allowed_{window}"] = _safe_mean(list(away_state.shots_against)[-window:])
            row[f"away_avg_sot_{window}"] = _safe_mean(list(away_state.shots_on_target_for)[-window:])
            row[f"away_avg_sot_allowed_{window}"] = _safe_mean(list(away_state.shots_on_target_against)[-window:])
            row[f"away_avg_points_{window}"] = _safe_mean(list(away_state.points)[-window:])

        row["goal_diff_form_5"] = row.get("home_avg_gf_5", 0.0) - row.get("away_avg_gf_5", 0.0)
        row["defensive_edge_5"] = row.get("away_avg_ga_5", 0.0) - row.get("home_avg_ga_5", 0.0)
        row["shots_edge_5"] = row.get("home_avg_shots_5", 0.0) - row.get("away_avg_shots_5", 0.0)
        row["points_edge_5"] = row.get("home_avg_points_5", 0.0) - row.get("away_avg_points_5", 0.0)

        if use_elo:
            row["home_elo"] = home_state.elo
            row["away_elo"] = away_state.elo
            row["elo_diff"] = home_state.elo - away_state.elo

        for odds_col in ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]:
            if odds_col in match.index:
                row[odds_col] = match.get(odds_col, np.nan)

        rows.append(row)

        home_goals = float(match.get("FTHG", 0.0) or 0.0)
        away_goals = float(match.get("FTAG", 0.0) or 0.0)
        home_shots = float(match.get("HS", 0.0) or 0.0)
        away_shots = float(match.get("AS", 0.0) or 0.0)
        home_sot = float(match.get("HST", 0.0) or 0.0)
        away_sot = float(match.get("AST", 0.0) or 0.0)

        home_state.gf.append(home_goals)
        home_state.ga.append(away_goals)
        home_state.shots_for.append(home_shots)
        home_state.shots_against.append(away_shots)
        home_state.shots_on_target_for.append(home_sot)
        home_state.shots_on_target_against.append(away_sot)
        home_state.points.append(_result_points(match["FTR"], "home"))

        away_state.gf.append(away_goals)
        away_state.ga.append(home_goals)
        away_state.shots_for.append(away_shots)
        away_state.shots_against.append(home_shots)
        away_state.shots_on_target_for.append(away_sot)
        away_state.shots_on_target_against.append(home_sot)
        away_state.points.append(_result_points(match["FTR"], "away"))

        if use_elo:
            exp_home, exp_away = _expected_score(home_state.elo, away_state.elo)
            actual_home = 1.0 if match["FTR"] == "H" else 0.5 if match["FTR"] == "D" else 0.0
            actual_away = 1.0 - actual_home
            k = 20.0
            home_state.elo += k * (actual_home - exp_home)
            away_state.elo += k * (actual_away - exp_away)

    feature_df = pd.DataFrame(rows)
    feature_df["target"] = feature_df["FTR"].map({"H": 0, "D": 1, "A": 2})
    return feature_df
