from __future__ import annotations

import pandas as pd

from src.backtest import run_backtest


def test_run_backtest_updates_bankroll_and_summary() -> None:
    signals = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-04-01"),
                "signal": "BET_H",
                "stake_fraction": 0.10,
                "pnl_multiple": 1.0,
            },
            {
                "Date": pd.Timestamp("2024-04-02"),
                "signal": "HOLD",
                "stake_fraction": 0.00,
                "pnl_multiple": 0.0,
            },
            {
                "Date": pd.Timestamp("2024-04-03"),
                "signal": "BET_A",
                "stake_fraction": 0.10,
                "pnl_multiple": -1.0,
            },
        ]
    )

    results, summary = run_backtest(signals, starting_bankroll=100.0)

    assert len(results) == 3
    assert round(summary["ending_bankroll"], 2) == 99.0
    assert round(summary["total_profit"], 2) == -1.0
    assert round(summary["total_staked"], 2) == 21.0
    assert summary["bets_placed"] == 2
    assert summary["bet_hit_rate"] == 0.5
