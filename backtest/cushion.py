"""
Honest path to 2 micros: a CUSHION-GATED size rule.
Trade 1 micro until realized profit reaches a cushion C; then switch to 2 micros.
Once you've banked C, a 2-micro drawdown has more room before the $2000 trailing MLL.

Question answered: what cushion C lets 2-micro trading run WITHOUT breaching the MLL,
out-of-sample? Models the trailing MLL exactly (peak-following, $2000 below peak equity).

Run: python -m backtest.cushion
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from backtest.data import load_cached, group_by_day
from backtest.sizing import replay_live_trades, _net_per_contract
from backtest.config import IN_SAMPLE_FRACTION, TRAILING_MLL


def simulate(per1, cushion, max_size=2):
    """Trade 1 micro until realized >= cushion, then max_size. Track trailing MLL
    breach. Returns (net, breach_trade_or_None, maxdd, final_size_reached)."""
    equity = 0.0; peak = 0.0; mll = -TRAILING_MLL
    realized = 0.0; size = 1; breach = None
    curve = []
    for k, base in enumerate(per1):
        if realized >= cushion:
            size = max_size
        pnl = base * size
        equity += pnl; realized += pnl
        if equity > peak:
            peak = equity; mll = peak - TRAILING_MLL
        curve.append(equity)
        if breach is None and equity < mll:
            breach = k + 1
    arr = np.array(curve)
    dd = float((arr - np.maximum.accumulate(arr)).min()) if len(arr) else 0.0
    return equity, breach, dd, size


def main():
    groups = group_by_day(load_cached())
    trades = replay_live_trades(groups)
    per1 = [_net_per_contract(t["pts"]) for t in trades]
    cut = int(len(per1) * IN_SAMPLE_FRACTION)
    oos = per1[cut:]

    print("="*74)
    print("  CUSHION-GATED 2-MICRO: trade 1 micro until +$C banked, then 2 micros")
    print("  Q: what cushion avoids the $2000 trailing-MLL breach? (OOS)")
    print("="*74)
    print(f"\n  {'cushion$':>9}{'scope':>6}{'net$':>9}{'maxDD$':>9}{'breach@':>9}")
    for cushion in (0, 250, 500, 750, 1000, 1500):
        for scope, seq in [("OOS", oos), ("full", per1)]:
            net, breach, dd, _ = simulate(seq, cushion)
            b = f"#{breach}" if breach else "survives"
            print(f"  {cushion:>9}{scope:>6}{net:>9.0f}{dd:>9.0f}{b:>9}")
        print()

    print("  Compare: flat 1 micro OOS net was +$1056 (maxDD -$1231, survives).")
    print("  A cushion 'works' only if it SURVIVES and nets more than flat 1 micro.")
    print("="*74)


if __name__ == "__main__":
    main()
