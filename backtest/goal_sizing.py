"""
GOAL-BASED SIZING (user's idea): set a per-period profit goal. Trade base size, but
as cumulative period P&L nears the goal, SHRINK size so a win doesn't overshoot; once
the goal is hit, STOP for the period. Tests: does this hit the goal more consistently,
and at what drawdown / MLL cost?

HONEST PRIORS:
 - Sizing cannot create expected profit on a no-edge signal (proven). It only reshapes
   the outcome distribution. Expect: higher hit-rate, but worse down-periods.
 - The "stop once goal hit" part is legit risk discipline (caps overtrading).
 - The "shrink near goal" part reduces overshoot + variance near the target only; the
   start-of-period full-size trades still carry the drawdown risk.

We measure vs FLAT (same base size, no goal logic):
 - % of periods that hit the goal
 - avg period P&L, and the WORST period
 - global max drawdown + $2000 trailing-MLL breach (eval survival)

Tested at multiple goals and intervals (weekly / daily) as requested.

Run: python -m backtest.goal_sizing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from backtest.data import load_cached, group_by_day
from backtest.sizing import replay_live_trades, _net_per_contract
from backtest.metrics import simulate_mll
from backtest.config import TRAILING_MLL

EST_WIN_PER_MICRO = 75.0   # causal estimate used to size-down near goal (avg hist win ~$76)
BASE_SIZE = 2              # base micros when far from goal
MAX_SIZE  = 2             # hard ceiling (eval safety)

def period_key(date_str, interval):
    d = pd.Timestamp(date_str)
    if interval == "weekly":
        iso = d.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return date_str  # daily

def run_goal(trades_net1, dates, goal, interval, use_goal_logic):
    """trades_net1 = per-trade net for 1 micro. Returns per-period totals + global net seq."""
    periods = {}
    for net1, d in zip(trades_net1, dates):
        periods.setdefault(period_key(d, interval), []).append(net1)

    period_totals = []
    global_seq = []   # the actual $ sequence (sized) for global drawdown/MLL
    for pk, seq in periods.items():
        cum = 0.0
        for net1 in seq:
            if use_goal_logic and cum >= goal:
                break  # goal hit -> stop for the period
            if use_goal_logic:
                gap = goal - cum
                # size so an average win lands near the goal; never exceed base/max
                want = max(1, int(np.ceil(gap / EST_WIN_PER_MICRO)))
                size = min(BASE_SIZE, MAX_SIZE, want)
            else:
                size = BASE_SIZE
            pnl = net1 * size
            cum += pnl
            global_seq.append(pnl)
        period_totals.append(cum)
    return period_totals, global_seq

def summarize(period_totals, global_seq, goal):
    pt = np.asarray(period_totals, float)
    g = np.asarray(global_seq, float)
    hit = (pt >= goal).mean() if len(pt) else 0
    dd = float((np.cumsum(g)-np.maximum.accumulate(np.cumsum(g))).min()) if len(g) else 0
    breach,_ = simulate_mll(list(g))
    return {
        "periods": len(pt), "hit_rate": hit, "avg": float(pt.mean()) if len(pt) else 0,
        "worst": float(pt.min()) if len(pt) else 0, "total": float(pt.sum()) if len(pt) else 0,
        "maxdd": dd, "breach": breach,
    }

def main():
    groups = group_by_day(load_cached())
    trades = replay_live_trades(groups)
    net1 = [_net_per_contract(t["pts"]) for t in trades]
    dates = [t["date"] for t in trades]
    print("="*86)
    print(f"  GOAL-BASED SIZING TEST  (base {BASE_SIZE} micros, ceiling {MAX_SIZE}, est win ${EST_WIN_PER_MICRO}/micro)")
    print("  Compares GOAL logic vs FLAT (same base size). Eval killer = $2000 MLL breach.")
    print("="*86)
    for interval in ["weekly", "daily"]:
        goals = [500, 1000, 1500] if interval=="weekly" else [200, 400]
        print(f"\n### INTERVAL = {interval} ###")
        print(f"  {'mode':<6}{'goal$':>6}{'periods':>8}{'hit%':>6}{'avg$':>8}{'worst$':>9}{'total$':>9}{'maxDD$':>9}{'breach':>8}")
        for goal in goals:
            for mode,use in [("FLAT",False),("GOAL",True)]:
                pt, gs = run_goal(net1, dates, goal, interval, use)
                s = summarize(pt, gs, goal)
                # FLAT hit-rate is vs same goal for comparison
                print(f"  {mode:<6}{goal:>6}{s['periods']:>8}{s['hit_rate']*100:>6.0f}"
                      f"{s['avg']:>8.0f}{s['worst']:>9.0f}{s['total']:>9.0f}{s['maxdd']:>9.0f}"
                      f"{('YES' if s['breach'] else 'no'):>8}")
    print("\n  Read: GOAL should HIT% higher than FLAT. But check WORST period + maxDD + breach.")
    print("  If GOAL hits more but worst/maxDD are worse or it breaches -> it just trades")
    print("  consistency for tail risk. On a no-edge signal, total$ can't beat FLAT's total$.")

if __name__=="__main__":
    main()
