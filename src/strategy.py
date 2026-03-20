from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_MAP = {0: "H", 1: "D", 2: "A"}
OUTCOME_TO_SUFFIX = {0: "H", 1: "D", 2: "A"}


def decimal_odds_to_implied_prob(odds: float) -> float:
    if pd.isna(odds) or odds <= 1.0:
        return np.nan
    return 1.0 / odds


def remove_overround(probs: np.ndarray) -> np.ndarray:
    probs = np.array(probs, dtype=float)
    total = np.nansum(probs)
    if total <= 0:
        return probs
    return probs / total


def kelly_fraction(prob: float, odds: float) -> float:
    if odds <= 1.0 or prob <= 0.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    frac = (b * prob - q) / b
    return max(frac, 0.0)


def generate_trade_signals(test_df: pd.DataFrame, proba: np.ndarray, config: dict) -> pd.DataFrame:
    bookmaker_prefix = config.get("bookmaker_prefix", "B365")
    edge_threshold = float(config.get("edge_threshold", 0.03))
    max_fractional_kelly = float(config.get("max_fractional_kelly", 0.25))
    max_bet_fraction = float(config.get("max_bet_fraction_of_bankroll", 0.05))
    min_odds = float(config.get("min_decimal_odds", 1.5))
    max_odds = float(config.get("max_decimal_odds", 6.0))

    rows: list[dict] = []
    for i, (_, match) in enumerate(test_df.iterrows()):
        odds_cols = [f"{bookmaker_prefix}H", f"{bookmaker_prefix}D", f"{bookmaker_prefix}A"]
        if not all(col in match.index for col in odds_cols):
            continue

        market_odds = np.array(
            [match[odds_cols[0]], match[odds_cols[1]], match[odds_cols[2]]],
            dtype=float,
        )
        if np.any(np.isnan(market_odds)):
            continue

        market_probs_raw = np.array([decimal_odds_to_implied_prob(o) for o in market_odds])
        market_probs = remove_overround(market_probs_raw)
        model_probs = proba[i]
        edges = model_probs - market_probs
        best_idx = int(np.argmax(edges))
        best_edge = float(edges[best_idx])
        chosen_odds = float(market_odds[best_idx])

        signal = "HOLD"
        stake_fraction = 0.0
        pnl_multiple = 0.0
        actual_target = int(match["target"])

        if best_edge >= edge_threshold and min_odds <= chosen_odds <= max_odds:
            signal = f"BET_{OUTCOME_MAP[best_idx]}"
            raw_kelly = kelly_fraction(float(model_probs[best_idx]), chosen_odds)
            stake_fraction = min(raw_kelly * max_fractional_kelly, max_bet_fraction)
            if actual_target == best_idx:
                pnl_multiple = (chosen_odds - 1.0)
            else:
                pnl_multiple = -1.0

        rows.append({
            "match_id": match["match_id"],
            "Date": match["Date"],
            "HomeTeam": match["HomeTeam"],
            "AwayTeam": match["AwayTeam"],
            "actual_result": match["FTR"],
            "model_home_prob": float(model_probs[0]),
            "model_draw_prob": float(model_probs[1]),
            "model_away_prob": float(model_probs[2]),
            "market_home_prob": float(market_probs[0]),
            "market_draw_prob": float(market_probs[1]),
            "market_away_prob": float(market_probs[2]),
            "edge_home": float(edges[0]),
            "edge_draw": float(edges[1]),
            "edge_away": float(edges[2]),
            "market_odds_home": float(market_odds[0]),
            "market_odds_draw": float(market_odds[1]),
            "market_odds_away": float(market_odds[2]),
            "signal": signal,
            "chosen_outcome": OUTCOME_MAP[best_idx] if signal != "HOLD" else "NONE",
            "chosen_odds": chosen_odds if signal != "HOLD" else np.nan,
            "stake_fraction": stake_fraction,
            "pnl_multiple": pnl_multiple,
            "best_edge": best_edge,
        })

    return pd.DataFrame(rows)
