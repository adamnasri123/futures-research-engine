"""
NOISE-AREA INTRADAY MOMENTUM — replication of Zarattini/Barbon/Aziz (SSRN 2024)
on our 520 days of 1-min ES, with our costs. An independent replication
(Quantitativo, Databento data, realistic futures costs) reported Sharpe ~0.9 on ES
2010-2025 — this tests whether the effect exists in OUR window (2024-06..2026-06).

FIXED RULES (declared before running; NO tuning; both exit variants pre-registered):
  - Noise area: for minute-of-day t, mu(t) = 14-day average of |C(d,t)/O(d) - 1|.
    UB(d,t) = max(Open_d, PrevClose_d) * (1 + mu(t))
    LB(d,t) = min(Open_d, PrevClose_d) * (1 - mu(t))
  - Long when a 1-min close is above UB; short when below LB (fill next bar open).
  - Reverse if a close crosses the OPPOSITE boundary.
  - Exit variant A ("vwap_stop"): also exit to flat when close crosses session VWAP
    against the position (re-entry allowed on a fresh boundary cross).
  - Exit variant B ("hold"): boundary reversals only, otherwise hold.
  - Entries 9:30-15:30, force-flat at 15:55 open. 1 micro, standard costs.

Run: python -m backtest.noise_area
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.structure_sar import load_1min
from backtest.costs import apply_costs
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS

RTH0, RTH1 = 9 * 60 + 30, 16 * 60
LAST_ENTRY, FLAT = 15 * 60 + 30, 15 * 60 + 55
LOOKBACK = 14


def prep(df):
    ts = df["timestamp"]
    df = df.assign(mins=ts.dt.hour * 60 + ts.dt.minute, date=ts.dt.date.astype(str))
    rth = df[(df.mins >= RTH0) & (df.mins < RTH1)].reset_index(drop=True)
    piv_c = rth.pivot_table(index="date", columns="mins", values="close")
    piv_o = rth.pivot_table(index="date", columns="mins", values="open")
    piv_h = rth.pivot_table(index="date", columns="mins", values="high")
    piv_l = rth.pivot_table(index="date", columns="mins", values="low")
    piv_v = rth.pivot_table(index="date", columns="mins", values="volume")
    opens = piv_o[RTH0]
    prev_close = piv_c[piv_c.columns.max()].shift(1)  # yesterday's 15:59 close
    move = (piv_c.div(opens, axis=0) - 1).abs()
    mu = move.rolling(LOOKBACK).mean().shift(1)       # causal
    return piv_o, piv_h, piv_l, piv_c, piv_v, opens, prev_close, mu


def run_variant(piv, variant):
    piv_o, piv_h, piv_l, piv_c, piv_v, opens, prev_close, mu = piv
    minutes = sorted(piv_c.columns)
    trades = []
    for d in piv_c.index:
        if pd.isna(prev_close.loc[d]) or mu.loc[d].isna().all():
            continue
        O, PC = opens.loc[d], prev_close.loc[d]
        if pd.isna(O) or pd.isna(PC):
            continue
        ub = max(O, PC) * (1 + mu.loc[d])
        lb = min(O, PC) * (1 - mu.loc[d])
        c = piv_c.loc[d]; o = piv_o.loc[d]
        h = piv_h.loc[d]; l = piv_l.loc[d]; v = piv_v.loc[d]
        tp = (h + l + c) / 3
        cum_pv = (tp * v).cumsum(); cum_v = v.cumsum()
        vwap = cum_pv / cum_v.replace(0, np.nan)

        pos = 0; entry = 0.0
        day_trades = []
        for i, t in enumerate(minutes[:-1]):
            ct = c.get(t); ubt = ub.get(t); lbt = lb.get(t)
            if pd.isna(ct) or pd.isna(ubt) or pd.isna(lbt):
                continue
            nxt = minutes[i + 1]
            ot_next = o.get(nxt)
            if pd.isna(ot_next):
                continue
            if t >= FLAT:
                break
            # force-flat at the 15:55 open
            if nxt >= FLAT and pos != 0:
                day_trades.append((entry, o.get(minutes[i + 1]), pos))
                pos = 0
                break
            if pos == 0:
                if t <= LAST_ENTRY:
                    if ct > ubt:
                        pos, entry = 1, ot_next
                    elif ct < lbt:
                        pos, entry = -1, ot_next
            else:
                vw = vwap.get(t)
                stop_hit = (variant == "vwap_stop" and not pd.isna(vw) and
                            ((pos > 0 and ct < vw) or (pos < 0 and ct > vw)))
                rev = (pos > 0 and ct < lbt) or (pos < 0 and ct > ubt)
                if rev:
                    day_trades.append((entry, ot_next, pos))
                    if t <= LAST_ENTRY:
                        pos, entry = -pos, ot_next
                    else:
                        pos = 0
                elif stop_hit:
                    day_trades.append((entry, ot_next, pos))
                    pos = 0
        if pos != 0:   # day ended with data gap
            last_c = c.dropna().iloc[-1]
            day_trades.append((entry, last_c, pos))
        for e, x, p in day_trades:
            pts = (x - e) * p
            trades.append({"date": d, "pts": pts,
                           "net": apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS)})
    return trades


def report(trades, label):
    a = np.array([t["net"] for t in trades])
    days = len(set(t["date"] for t in trades))
    if len(a) == 0:
        print(f"  {label}: no trades"); return
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = wins.sum() / gl if gl > 0 else float("inf")
    eq = np.cumsum(a); dd = (eq - np.maximum.accumulate(eq)).min()
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std(ddof=1) > 0 else 0
    print(f"  {label:<12} n={len(a):>5} ({len(a)/max(1,days):.1f}/day)  net ${a.sum():>8.0f}  "
          f"exp ${a.mean():>6.2f}  PF {pf:.2f}  win {(a>0).mean()*100:.0f}%  "
          f"maxDD ${dd:>7.0f}  t={t:+.2f}")
    # yearly
    ys = {}
    for tr in trades:
        ys.setdefault(tr["date"][:4], []).append(tr["net"])
    for y in sorted(ys):
        b = np.array(ys[y])
        print(f"      {y}: n={len(b):>4}  net ${b.sum():>7.0f}  exp ${b.mean():>6.2f}")


def main():
    df = load_1min()
    piv = prep(df)
    print("=" * 92)
    print("  NOISE-AREA INTRADAY MOMENTUM (Zarattini et al. rules, fixed; 1 micro; our costs)")
    print("=" * 92)
    for variant in ("vwap_stop", "hold"):
        trades = run_variant(piv, variant)
        report(trades, variant)
    print("\n  Read: independent replication got Sharpe ~0.9 on 2010-2025. If our 2-year")
    print("  window is flat/negative, that's consistent with regime-dependence (their")
    print("  edge concentrates in high-vol years) — log it, don't extrapolate either way.")


if __name__ == "__main__":
    main()
