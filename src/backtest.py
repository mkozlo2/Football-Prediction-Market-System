from __future__ import annotations

import pandas as pd


def run_backtest(signals: pd.DataFrame, starting_bankroll: float) -> tuple[pd.DataFrame, dict[str, float]]:
    bankroll = float(starting_bankroll)
    rows: list[dict] = []
    placed_bets = 0
    wins = 0

    signals = signals.sort_values("Date").reset_index(drop=True).copy()

    for _, row in signals.iterrows():
        stake_amount = bankroll * float(row.get("stake_fraction", 0.0))
        pnl = 0.0
        if row["signal"] != "HOLD" and stake_amount > 0:
            placed_bets += 1
            pnl = stake_amount * float(row["pnl_multiple"])
            if pnl > 0:
                wins += 1
            bankroll += pnl

        rows.append({
            **row.to_dict(),
            "stake_amount": stake_amount,
            "trade_pnl": pnl,
            "bankroll_after": bankroll,
        })

    results = pd.DataFrame(rows)
    total_profit = bankroll - starting_bankroll
    total_staked = float(results["stake_amount"].sum()) if not results.empty else 0.0
    roi = (total_profit / total_staked) if total_staked > 0 else 0.0
    hit_rate = (wins / placed_bets) if placed_bets > 0 else 0.0

    summary = {
        "starting_bankroll": float(starting_bankroll),
        "ending_bankroll": float(bankroll),
        "total_profit": float(total_profit),
        "total_staked": float(total_staked),
        "roi": float(roi),
        "bets_placed": int(placed_bets),
        "bet_hit_rate": float(hit_rate),
    }
    return results, summary
