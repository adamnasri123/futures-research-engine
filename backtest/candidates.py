"""
CANDIDATE STRATEGIES from the Market Atlas (2026-06-11) — focused honest tests.

Candidate 1 — GAP FADE (atlas study C): overnight gap (prior RTH close -> open) of
10-25 pts faded at the 9:30 open; intraday follow-through against the gap averaged
+7 pts (t=2.31), small gaps filled 90% of days. Test: enter at 9:35 open against the
gap; target = prior close (the fill); variants of stop; flat 15:55.

Candidate 2 — HOUR MOMENTUM (atlas study D): 60-min serial correlation +0.05
(t=8.2). Test: at each half-hour 10:30-14:30, if |last 60m return| >= theta * sigma60
(rolling, causal), trade its direction for exactly 60 minutes. First signal per day.

Both: 1 micro, full costs, 70/30 train/OOS, random-direction control on the same
entries. PROMOTION requires OOS holding up AND beating the control. n is ~150-500 —
even a pass is "paper-trade forward", not live.

Run: python -m backtest.candidates
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.atlas import load, daily_frame, tstat
from backtest.costs import apply_costs
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION

RTH_START, RTH_END = 9 * 60 + 30, 16 * 60
FLAT_MIN = 15 * 60 + 55


def stats_row(nets, label):
    a = np.asarray(nets, float)
    if len(a) == 0:
        return f"  {label:<30} (no trades)"
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = wins.sum() / gl if gl > 0 else float("inf")
    eq = np.cumsum(a); dd = float((eq - np.maximum.accumulate(eq)).min())
    return (f"  {label:<30}{len(a):>5}{a.sum():>9.0f}{a.mean():>8.2f}{pf:>6.2f}"
            f"{(a > 0).mean()*100:>5.0f}{dd:>9.0f}{tstat(a):>7.2f}")


HDR = f"  {'variant':<30}{'n':>5}{'net$':>9}{'exp$':>8}{'PF':>6}{'win%':>5}{'maxDD$':>9}{'t':>7}"


# --------------------------------------------------------------------------- #
def gap_fade(df, d, lo=10, hi=25, stop_mode="none"):
    """Fade the overnight gap at the 9:35 open; target = prior close; flat 15:55.
    stop_mode: none | gap (stop = gap extension by its own size) | fixed25."""
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    trades = []
    for date, day in rth.groupby("date"):
        if date not in d.index:
            continue
        row = d.loc[date]
        gap = row.gap
        if pd.isna(gap) or not (lo <= abs(gap) <= hi):
            continue
        if len(day) < 10:
            continue
        side = -np.sign(gap)                  # fade: gap up -> short
        # enter at the open of the 9:35 bar (second RTH bar - lets the open print)
        entry = day.open.iloc[1]
        target = row.prev_close
        if stop_mode == "gap":
            stop = entry + side * (-abs(gap))     # adverse move = one more gap-size
        elif stop_mode == "fixed25":
            stop = entry - side * 25
        else:
            stop = None
        exit_px = day.close.iloc[-1]
        for j in range(1, len(day)):
            hj, lj, mj = day.high.iloc[j], day.low.iloc[j], day.mins.iloc[j]
            if mj >= FLAT_MIN:
                exit_px = day.open.iloc[j]; break
            if stop is not None and ((side > 0 and lj <= stop) or (side < 0 and hj >= stop)):
                exit_px = stop; break
            if (side > 0 and hj >= target) or (side < 0 and lj <= target):
                exit_px = target; break
        pts = (exit_px - entry) * side
        trades.append({"date": str(date), "pts": pts,
                       "net": apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS),
                       "side": side})
    return trades


def hour_momentum(df, theta=1.0):
    """At 10:30/11:00/.../14:30: signal = last 60m return; if |signal| >= theta*sigma60
    (rolling 30-day causal), hold its direction for 60 minutes. First signal/day."""
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)].reset_index(drop=True)
    c = rth.close.to_numpy(); o = rth.open.to_numpy()
    mins = rth.mins.to_numpy(); dates = rth.date.astype(str).to_numpy()

    # rolling sigma of 60-min returns, causal (computed from prior 30 trading days)
    r60 = pd.Series(c).diff(12).to_numpy()
    sig = pd.Series(r60).rolling(12 * 78 // 2).std().shift(1).to_numpy()  # ~30d of bars

    trades = []
    done_day = None
    checkpoints = {10*60+30, 11*60, 11*60+30, 12*60, 12*60+30, 13*60, 13*60+30, 14*60, 14*60+30}
    for i in range(13, len(c) - 13):
        if dates[i] == done_day or mins[i] not in checkpoints:
            continue
        if dates[i] != dates[i - 12] or np.isnan(sig[i]) or sig[i] == 0:
            continue
        s = r60[i]
        if abs(s) < theta * sig[i]:
            continue
        side = 1 if s > 0 else -1
        entry = o[i + 1]
        # exit after 12 bars or at day end
        jx = i + 1 + 12
        if jx >= len(c) or dates[jx] != dates[i]:
            day_end = np.max(np.nonzero(dates == dates[i])[0])
            jx = day_end
        pts = (c[jx] - entry) * side
        trades.append({"date": dates[i], "pts": pts,
                       "net": apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS),
                       "side": side})
        done_day = dates[i]
    return trades


def random_control(trades, runs=500, seed=3):
    """Same entries, random sign: returns p95 of total net (cost-adjusted)."""
    rng = np.random.default_rng(seed)
    pts = np.array([t["pts"] * (1 if t["side"] > 0 else -1) * t["side"] for t in trades])
    # reconstruct unsigned move: pts_signed relative to side -> raw move = pts*side... simpler:
    raw = np.array([t["pts"] for t in trades])     # already side-adjusted P&L in pts
    # random sign flip around zero: flip half the trades' sign
    totals = []
    for _ in range(runs):
        fl = rng.choice([-1, 1], size=len(raw))
        nets = [apply_costs(p * f * POINT_VALUE, SLIPPAGE_TICKS) for p, f in zip(raw, fl)]
        totals.append(sum(nets))
    return np.percentile(totals, 95), np.mean(totals)


def report(trades, name):
    print(f"\n### {name}")
    print(HDR)
    nets = [t["net"] for t in trades]
    cut = int(len(nets) * IN_SAMPLE_FRACTION)
    print(stats_row(nets, "full"))
    print(stats_row(nets[:cut], "train (first 70%)"))
    print(stats_row(nets[cut:], "OOS (last 30%)"))
    if len(trades) >= 30:
        p95, mu = random_control(trades)
        print(f"  random-sign control: mean total ${mu:.0f}, p95 ${p95:.0f}  "
              f"(strategy full total must beat p95)")


def main():
    df = load()
    d = daily_frame(df)
    print(f"data: {len(d)} days")

    print("\n" + "=" * 92)
    print("  CANDIDATE 1 — GAP FADE (from atlas C)")
    print("=" * 92)
    for lo, hi, sm in [(10, 25, "none"), (10, 25, "gap"), (10, 25, "fixed25"),
                       (10, 60, "none"), (25, 60, "none")]:
        trades = gap_fade(df, d, lo, hi, sm)
        report(trades, f"gap {lo}-{hi} pts, stop={sm}")

    print("\n" + "=" * 92)
    print("  CANDIDATE 2 — HOUR MOMENTUM (from atlas D)")
    print("=" * 92)
    for theta in (0.5, 1.0, 1.5):
        trades = hour_momentum(df, theta)
        report(trades, f"theta={theta} sigma")

    print("\n  READ: a candidate is PROMOTED to paper trading only if OOS sign matches")
    print("  train, expectancy clears costs with margin, and full total > control p95.")
    print("  Even then: paper-forward first. n here is small; respect that.")


if __name__ == "__main__":
    main()
