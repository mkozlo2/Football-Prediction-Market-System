from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import generate_trade_signals, remove_overround


def test_remove_overround_normalizes_probabilities() -> None:
    probs = np.array([0.55, 0.30, 0.25])

    normalized = remove_overround(probs)

    assert np.isclose(normalized.sum(), 1.0)
    assert all(value > 0 for value in normalized)


def test_generate_trade_signals_places_bet_on_positive_edge() -> None:
    test_df = pd.DataFrame(
        [
            {
                "match_id": "10",
                "Date": pd.Timestamp("2024-03-01"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTR": "H",
                "target": 0,
                "B365H": 2.5,
                "B365D": 3.2,
                "B365A": 3.0,
            }
        ]
    )
    proba = np.array([[0.55, 0.20, 0.25]])
    config = {
        "bookmaker_prefix": "B365",
        "edge_threshold": 0.03,
        "max_fractional_kelly": 0.25,
        "max_bet_fraction_of_bankroll": 0.05,
        "min_decimal_odds": 1.5,
        "max_decimal_odds": 6.0,
    }

    signals = generate_trade_signals(test_df, proba, config)

    assert len(signals) == 1
    assert signals.loc[0, "signal"] == "BET_H"
    assert signals.loc[0, "chosen_outcome"] == "H"
    assert signals.loc[0, "stake_fraction"] > 0
    assert signals.loc[0, "pnl_multiple"] == 1.5


def test_generate_trade_signals_holds_when_edge_is_too_small() -> None:
    test_df = pd.DataFrame(
        [
            {
                "match_id": "11",
                "Date": pd.Timestamp("2024-03-02"),
                "HomeTeam": "Alpha",
                "AwayTeam": "Gamma",
                "FTR": "D",
                "target": 1,
                "B365H": 2.0,
                "B365D": 3.4,
                "B365A": 4.0,
            }
        ]
    )
    proba = np.array([[0.50, 0.28, 0.22]])
    config = {
        "bookmaker_prefix": "B365",
        "edge_threshold": 0.20,
        "max_fractional_kelly": 0.25,
        "max_bet_fraction_of_bankroll": 0.05,
        "min_decimal_odds": 1.5,
        "max_decimal_odds": 6.0,
    }

    signals = generate_trade_signals(test_df, proba, config)

    assert signals.loc[0, "signal"] == "HOLD"
    assert signals.loc[0, "stake_fraction"] == 0.0
