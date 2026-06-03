"""
Liquidity-aware strategy layer + mega-sweep with multiple-testing correction.

THE INSIGHT (user's): price is drawn to LIQUIDITY POOLS — clusters of resting stop
orders — because that's where size can get filled. Pools sit at obvious levels:
prior-day high/low, opening-range high/low, and prior swing highs/lows (equal highs/
lows). Two tradeable mechanisms:
  - SWEEP-REVERSAL ("stop hunt"): price spikes THROUGH a pool, fails to hold, and
    reverses. Enter the reversal. (Fade the sweep.)
  - LIQUIDITY-TARGET: exit by riding toward the nearest un-swept pool (that's where
    price "wants" to go), instead of a fixed R multiple.

DATA LIMITATION (honest): cache is RTH-only (9:30-15:55). No overnight/Globex, so the
true overnight high/low pools are MISSING. We use prior-RTH-day levels + opening range
+ swing pools. If this shows promise, overnight data is the next step.

HONESTY ENGINE — White's Reality Check (multiple-testing correction):
  Testing K strategies and keeping the best GUARANTEES a lucky winner. So the bar
  scales with K: the best real strategy must beat the 95th percentile of the
  BEST-OF-K *random* strategies. Try more combos -> harsher bar. This is the only
  way to search broadly without fooling ourselves. We accept the verdict, incl NO-GO.

Run: python -m backtest.liquidity
"""
import sys
import random
from pathlib import Path
from dataclasses import dataclass
from itertools import product
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, TICK_SIZE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION

# ---- params ----
ATR_PERIOD   = 14
SWING_K      = 2
OR_BARS      = 6          # opening range = first 6 5-min bars (30 min)
SWEEP_TOL_ATR = 0.10      # how far beyond a pool counts as a "sweep" poke
# NOTE: ctx.mins is MINUTES SINCE 9:30 OPEN (0..385), not minutes-of-day.
ENTRY_START  = 5          # 9:35 ET
ENTRY_END    = 150        # 12:00 ET
FLAT_MIN     = 385        # 15:55 ET


def _atr(h, l, c, period):
    n = len(h); tr = np.empty(n); tr[0] = h[0]-l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.full(n, np.nan)
    if n >= period:
        a[period-1] = tr[:period].mean()
        for i in range(period, n):
            a[i] = (a[i-1]*(period-1)+tr[i])/period
    return a


def _ema(arr, p):
    out = np.full(len(arr), np.nan)
    if not len(arr): return out
    k = 2/(p+1); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i]*k + out[i-1]*(1-k)
    return out


def _swings(h, l, k):
    n = len(h); sh, sl = [], []
    for i in range(k, n-k):
        if h[i] == h[i-k:i+k+1].max() and h[i] > h[i-1] and h[i] > h[i+1]:
            sh.append((i, h[i], i+k))
        if l[i] == l[i-k:i+k+1].min() and l[i] < l[i-1] and l[i] < l[i+1]:
            sl.append((i, l[i], i+k))
    return sh, sl


@dataclass(frozen=True)
class LParams:
    entry: str       # 'sweep_rev', 'sweep_rev_pd', 'breakout' (control)
    exit:  str       # 'fixed_2r', 'liq_target', 'swing_trail', 'chandelier'
    stop_atr: float
    def label(self):
        return f"{self.entry}->{self.exit} stop{self.stop_atr}"


class LCtx:
    """Per-day context with liquidity pools. Causal: prior-day levels known at open;
    opening range known after OR_BARS; swings only after K-bar confirm."""
    __slots__ = ("o","h","l","c","v","mins","atr","ema","sh","sl",
                 "pdh","pdl","orh","orl","n")
    def __init__(self, day5m, prev_hi, prev_lo):
        self.o = day5m["open"].to_numpy(); self.h = day5m["high"].to_numpy()
        self.l = day5m["low"].to_numpy();  self.c = day5m["close"].to_numpy()
        self.v = day5m["volume"].to_numpy()
        ts = day5m["timestamp"]
        self.mins = ((ts.dt.hour*60+ts.dt.minute)-(9*60+30)).to_numpy()
        self.n = len(day5m)
        self.atr = _atr(self.h, self.l, self.c, ATR_PERIOD)
        self.ema = _ema(self.c, 10)
        self.sh, self.sl = _swings(self.h, self.l, SWING_K)
        self.pdh, self.pdl = prev_hi, prev_lo     # prior-day high/low (pools)
        # opening range pools (known only after OR_BARS bars)
        if self.n > OR_BARS:
            self.orh = self.h[:OR_BARS].max()
            self.orl = self.l[:OR_BARS].min()
        else:
            self.orh = self.orl = None

    def pools_above(self, i, px):
        out = []
        if self.pdh and self.pdh > px: out.append(self.pdh)
        if self.orh and i >= OR_BARS and self.orh > px: out.append(self.orh)
        for (idx, p, conf) in self.sh:
            if conf <= i and p > px: out.append(p)
        return sorted(out)

    def pools_below(self, i, px):
        out = []
        if self.pdl and self.pdl < px: out.append(self.pdl)
        if self.orl and i >= OR_BARS and self.orl < px: out.append(self.orl)
        for (idx, p, conf) in self.sl:
            if conf <= i and p < px: out.append(p)
        return sorted(out, reverse=True)


def _entry(ctx, i, method):
    """Return 'long'/'short'/None. Sweep-reversal = fade a failed poke through a pool."""
    if np.isnan(ctx.atr[i]) or i < 1:
        return None
    tol = SWEEP_TOL_ATR * ctx.atr[i]
    h, l, c, o = ctx.h, ctx.l, ctx.c, ctx.o

    # collect candidate pools depending on method
    if method in ("sweep_rev", "sweep_rev_pd"):
        # pools just below (for long sweep) and above (for short sweep)
        below = [ctx.pdl] if method == "sweep_rev_pd" else ctx.pools_below(i, c[i])
        above = [ctx.pdh] if method == "sweep_rev_pd" else ctx.pools_above(i, c[i])
        below = [x for x in below if x]
        above = [x for x in above if x]
        # LONG: this bar's low pierced a pool below but CLOSED back above it (failed sweep)
        for pool in below:
            if l[i] < pool - tol and c[i] > pool:
                return "long"
        # SHORT: this bar's high pierced a pool above but CLOSED back below it
        for pool in above:
            if h[i] > pool + tol and c[i] < pool:
                return "short"
        return None

    if method == "breakout":   # control: plain 6-bar breakout (no liquidity)
        if i < 6: return None
        e = ctx.ema[i]
        if np.isnan(e): return None
        if c[i] > e and e > ctx.ema[i-1] and c[i] > h[i-6:i].max(): return "long"
        if c[i] < e and e < ctx.ema[i-1] and c[i] < l[i-6:i].min(): return "short"
        return None
    return None


def _manage(ctx, entry_idx, side, p: LParams):
    h, l, c, atr = ctx.h, ctx.l, ctx.c, ctx.atr
    n = ctx.n
    entry = ctx.o[entry_idx]
    a0 = atr[entry_idx-1] if not np.isnan(atr[entry_idx-1]) else atr[entry_idx]
    stop = entry - p.stop_atr*a0 if side == "long" else entry + p.stop_atr*a0
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    # target
    if p.exit == "fixed_2r":
        tgt = entry + 2*risk if side == "long" else entry - 2*risk
    elif p.exit == "liq_target":
        # nearest opposing pool = where liquidity sits
        if side == "long":
            pa = ctx.pools_above(entry_idx, entry)
            tgt = pa[0] if pa else entry + 2*risk
        else:
            pb = ctx.pools_below(entry_idx, entry)
            tgt = pb[0] if pb else entry - 2*risk
    else:
        tgt = None  # trailing-only exits

    hh = entry
    for i in range(entry_idx, n):
        hh = max(hh, h[i]) if side == "long" else min(hh, l[i])
        if p.exit == "swing_trail":
            if side == "long":
                sw = max([px for (idx,px,conf) in ctx.sl if conf <= i and px < c[i]], default=None)
                if sw: stop = max(stop, sw)
            else:
                sw = min([px for (idx,px,conf) in ctx.sh if conf <= i and px > c[i]], default=None)
                if sw: stop = min(stop, sw)
        elif p.exit == "chandelier":
            stop = max(stop, hh - p.stop_atr*atr[i]) if side == "long" else min(stop, hh + p.stop_atr*atr[i])

        if side == "long" and l[i] <= stop: return (stop-entry)*POINT_VALUE, "stop"
        if side == "short" and h[i] >= stop: return (entry-stop)*POINT_VALUE, "stop"
        if tgt is not None:
            if side == "long" and h[i] >= tgt: return (tgt-entry)*POINT_VALUE, "target"
            if side == "short" and l[i] <= tgt: return (entry-tgt)*POINT_VALUE, "target"
        if ctx.mins[i] >= FLAT_MIN:
            px = c[i]
            return ((px-entry) if side=="long" else (entry-px))*POINT_VALUE, "time"
    px = c[-1]
    return ((px-entry) if side=="long" else (entry-px))*POINT_VALUE, "session_end"


def build_contexts(groups):
    ctxs = []
    prev_hi = prev_lo = None
    for date, day in groups:
        ctx = LCtx(day, prev_hi, prev_lo)
        ctxs.append((date, ctx))
        prev_hi = float(day["high"].max()); prev_lo = float(day["low"].min())
    return ctxs


def run_one(ctx, p: LParams):
    """One trade/day. Returns net $ for 1 contract, or None."""
    for i in range(ATR_PERIOD+1, ctx.n-1):
        if ctx.mins[i] < ENTRY_START or ctx.mins[i] > ENTRY_END:
            continue
        side = _entry(ctx, i, p.entry)
        if side is None:
            continue
        res = _manage(ctx, i+1, side, p)
        if res is None:
            return None
        return apply_costs(res[0], SLIPPAGE_TICKS)
    return None


def run_combo(ctxs, p, dates_set):
    nets = []
    for date, ctx in ctxs:
        if date not in dates_set:
            continue
        r = run_one(ctx, p)
        if r is not None:
            nets.append(r)
    return nets


def stats(nets):
    n = len(nets)
    if n == 0: return None
    a = np.asarray(nets, float)
    wins = a[a>0]; gl = -a[a<=0].sum()
    pf = (wins.sum()/gl) if gl > 0 else float("inf")
    breach, _ = simulate_mll(list(a))
    return {"n": n, "net": float(a.sum()), "exp": float(a.mean()),
            "pf": pf, "wr": len(wins)/n, "breach": breach}


def main():
    random.seed(1)
    print("="*72)
    print("  LIQUIDITY SWEEP + White's Reality Check (multiple-testing correction)")
    print("="*72)
    groups = group_by_day(load_cached())
    ctxs = build_contexts(groups)
    dates = [d for d, _ in groups]
    cut = int(len(dates) * (IN_SAMPLE_FRACTION + 0.0))
    # 3-way: train 50 / val 20 / holdout 30
    i_tr = int(len(dates)*0.5); i_va = int(len(dates)*0.7)
    train, val, hold = set(dates[:i_tr]), set(dates[i_tr:i_va]), set(dates[i_va:])
    print(f"\n{len(dates)} days | train {len(train)} val {len(val)} holdout {len(hold)}")
    print("DATA NOTE: RTH-only — no overnight liquidity pools (limitation).")

    entries = ["sweep_rev", "sweep_rev_pd", "breakout"]
    exits   = ["fixed_2r", "liq_target", "swing_trail", "chandelier"]
    stops   = [1.0, 1.5, 2.0, 2.5, 3.0]
    grid = [LParams(e, x, s) for e, x, s in product(entries, exits, stops)]
    K = len(grid)
    print(f"\nGrid: {K} strategies ({len(entries)} entries x {len(exits)} exits x {len(stops)} stops)")

    # rank on TRAIN
    rows = []
    for p in grid:
        s = stats(run_combo(ctxs, p, train))
        if s and s["n"] >= 25:
            rows.append((p, s))
    rows.sort(key=lambda r: r[1]["pf"], reverse=True)
    print(f"\nTop 8 on TRAIN (of {len(rows)} with >=25 trades):")
    print(f"  {'strategy':<34}{'n':>4}{'PF':>6}{'net$':>8}{'win%':>6}")
    for p, s in rows[:8]:
        print(f"  {p.label():<34}{s['n']:>4}{s['pf']:>6.2f}{s['net']:>8.0f}{s['wr']*100:>6.0f}")

    # validation survivors
    cands = []
    for p, tr in rows[:15]:
        if tr["pf"] <= 1.1: continue
        va = stats(run_combo(ctxs, p, val))
        if va and va["n"] >= 12 and va["pf"] > 1.1 and va["pf"] >= 0.6*tr["pf"]:
            cands.append((p, tr, va))
    print(f"\nSurvived validation: {len(cands)}")
    for p, tr, va in cands[:8]:
        print(f"  {p.label():<34} trPF {tr['pf']:.2f}  vaPF {va['pf']:.2f}")

    if not cands:
        print("\nNO candidate generalized train->val. VERDICT: NO-GO. (Expected.)")
        return

    # --- White's Reality Check on HOLDOUT ---
    # Best candidate's holdout net vs the 95th pct of best-of-K RANDOM strategies.
    print(f"\n--- HOLDOUT + Reality Check (bar scales with K={K}) ---")
    # precompute per-day random outcomes on holdout: for each day, the net of entering
    # random side at a random eligible bar with a representative exit (fixed_2r, stop 2.0)
    proxy = LParams("breakout", "fixed_2r", 2.0)
    per_day_rand = []
    for date, ctx in ctxs:
        if date not in hold: continue
        outs = []
        for i in range(ATR_PERIOD+1, ctx.n-1):
            if ctx.mins[i] < ENTRY_START or ctx.mins[i] > ENTRY_END: continue
            if np.isnan(ctx.atr[i]): continue
            for sd in ("long","short"):
                r = _manage(ctx, i+1, sd, proxy)
                if r is not None:
                    outs.append(apply_costs(r[0], SLIPPAGE_TICKS))
        if outs: per_day_rand.append(outs)

    def random_strategy_net():
        return sum(random.choice(o) for o in per_day_rand)

    NSIM = 2000
    best_of_k = np.empty(NSIM)
    for s_ in range(NSIM):
        best_of_k[s_] = max(random_strategy_net() for _ in range(K))
    bar = float(np.percentile(best_of_k, 95))

    best_p, _, _ = cands[0]
    ho = stats(run_combo(ctxs, best_p, hold))
    print(f"  best candidate: {best_p.label()}")
    print(f"  holdout: n={ho['n']} PF={ho['pf']:.2f} net=${ho['net']:.0f} "
          f"win={ho['wr']*100:.0f}% MLLbreach={'YES' if ho['breach'] else 'no'}")
    print(f"  Reality-Check bar (95th pct of best-of-{K} random): ${bar:.0f}")
    passed = ho["net"] > bar and not ho["breach"] and ho["pf"] > 1.3
    print(f"\n  VERDICT: {'GO (beat the corrected bar!)' if passed else 'NO-GO'}")
    if not passed:
        print("  The survivor did NOT beat the multiple-testing-corrected random bar.")
        print("  This is the honest result: searching harder did not find real edge.")


if __name__ == "__main__":
    main()
