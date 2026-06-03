"""
Does adding REGIME or NEWS information make a robust edge appear?

Re-runs the exact same honest pipeline as sweep.py (train -> validation ->
sacred holdout + random benchmark + walk-forward), but the strategy only ACTS on
days permitted by a day-filter. The full calendar is preserved (the bot "shows up"
every day, just stands aside on disallowed days), so walk-forward stays valid.

Filters (all pre-committed, standard thresholds — no fitting):
  baseline       : trade every day (reproduces the prior NO-GO)
  trend_only     : only days the daily ADX/Choppiness flags as TRENDING
  chop_only      : only CHOP days (sanity check — trend strat should do worse)
  skip_news      : skip FOMC + NFP days
  trend_skipnews : TRENDING days that are NOT news days

Run: python -m backtest.regime_news
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.metrics import simulate_mll
from backtest.sweep import build_grid, build_ctx_cache
from backtest.modular import run_ctx, random_outcomes_ctx
from backtest.costs import apply_costs
from backtest.config import (
    SWEEP_TRAIN_FRAC, SWEEP_VAL_FRAC, SWEEP_MIN_TRADES,
    SLIPPAGE_TICKS, SLIPPAGE_STRESS_TICKS,
)
from backtest.regime import classify
from backtest.news import tag_days

OUT = open("backtest/regime_news_results.txt", "w")
def log(msg=""):
    print(msg)
    OUT.write(msg + "\n"); OUT.flush()


def stats(nets):
    n = len(nets)
    if n == 0:
        return None
    a = np.asarray(nets, float)
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    return {"n": n, "pf": pf, "net": float(a.sum()), "exp": float(a.mean()),
            "wr": len(wins) / n}


def combo_nets(dates, cache, p, allowed, slippage=SLIPPAGE_TICKS):
    nets = []
    for d in dates:
        if d not in allowed:
            continue
        t = run_ctx(cache[(d, p.exec_tf, p.trend_tf)], d, p)
        if t is not None:
            nets.append(apply_costs(t.gross_pnl, slippage))
    return nets


def bench(dates, cache, p, allowed, runs=400):
    per_day = []
    for d in dates:
        if d not in allowed:
            continue
        outs = random_outcomes_ctx(cache[(d, p.exec_tf, p.trend_tf)], p)
        if outs:
            per_day.append(outs)
    if not per_day:
        return None
    tot = np.empty(runs)
    for r in range(runs):
        tot[r] = sum(random.choice(o) for o in per_day)
    tot.sort()
    return {"p95": float(tot[int(runs*0.95)]), "mean": float(tot.mean())}


def pipeline(name, dates, train, val, hold, cache, grid, allowed):
    # TRAIN: rank
    rows = []
    for p in grid:
        s = stats(combo_nets(train, cache, p, allowed))
        if s and s["n"] >= max(20, SWEEP_MIN_TRADES * 0.5):
            rows.append((p, s))
    rows.sort(key=lambda r: r[1]["pf"], reverse=True)
    if not rows:
        log(f"  [{name}] no combos with enough trades; skipping.")
        return None

    # VALIDATION: keep generalizers
    cands = []
    for p, tr in rows[:30]:
        if tr["pf"] <= 1.1:
            continue
        va = stats(combo_nets(val, cache, p, allowed))
        if va and va["n"] >= 12 and va["pf"] > 1.1 and va["pf"] >= 0.6 * tr["pf"]:
            cands.append((p, tr, va))
    cands.sort(key=lambda r: r[2]["pf"], reverse=True)

    if not cands:
        log(f"  [{name}] survivors after validation: 0  ->  NO-GO")
        return {"go": False, "best": None}

    # HOLDOUT: grade survivors once
    best = None
    any_go = False
    for p, tr, va in cands[:5]:
        ho = stats(combo_nets(hold, cache, p, allowed))
        if ho is None:
            continue
        st = stats(combo_nets(hold, cache, p, allowed, slippage=SLIPPAGE_STRESS_TICKS))
        bm = bench(hold, cache, p, allowed)
        breach, _ = simulate_mll(combo_nets(hold, cache, p, allowed))
        beats = bm is not None and ho["net"] > bm["p95"]
        gates_ok = (ho["n"] >= 30 and ho["pf"] > 1.3 and beats
                    and not breach and st is not None and st["pf"] > 1.0)
        any_go = any_go or gates_ok
        if best is None or ho["pf"] > best[1]["pf"]:
            best = (p, ho, st, bm, beats, breach, gates_ok)

    p, ho, st, bm, beats, breach, gates_ok = best
    log(f"  [{name}] survivors={len(cands)}  best holdout: {p.label()}")
    log(f"      n={ho['n']} PF={ho['pf']:.2f} net=${ho['net']:.0f} win={ho['wr']*100:.0f}%"
        f"  stressPF={st['pf']:.2f}" if st else "")
    if bm:
        log(f"      random p95=${bm['p95']:.0f}  beats={'Y' if beats else 'N'}  "
            f"MLLbreach={'Y' if breach else 'N'}  -> {'GO' if gates_ok else 'NO-GO'}")
    return {"go": any_go, "best": best}


def walk_forward(name, dates, cache, grid, allowed):
    n = len(dates)
    tr_w, te_w, step = 150, 50, 50
    nets = []
    start = 0
    while start + tr_w + te_w <= n:
        tr = dates[start:start+tr_w]
        te = dates[start+tr_w:start+tr_w+te_w]
        best, bpf = None, -1
        for p in grid:
            s = stats(combo_nets(tr, cache, p, allowed))
            if s and s["n"] >= 20 and s["pf"] > bpf:
                best, bpf = p, s["pf"]
        if best is not None:
            nets.extend(combo_nets(te, cache, best, allowed))
        start += step
    wf = stats(nets)
    if wf:
        breach, _ = simulate_mll(nets)
        log(f"  [{name}] walk-forward: n={wf['n']} PF={wf['pf']:.2f} "
            f"net=${wf['net']:.0f} exp=${wf['exp']:.2f} MLLbreach={'Y' if breach else 'N'}"
            f"  -> {'WF-EDGE' if wf['pf']>1.3 else 'no edge'}")
    else:
        log(f"  [{name}] walk-forward: no trades")


def main():
    random.seed(42)
    g = group_by_day(load_cached())
    dates = [d for d, _ in g]
    n = len(dates)
    i_tr = int(n * SWEEP_TRAIN_FRAC)
    i_va = int(n * (SWEEP_TRAIN_FRAC + SWEEP_VAL_FRAC))
    train, val, hold = dates[:i_tr], dates[i_tr:i_va], dates[i_va:]

    log("=" * 64)
    log("  REGIME + NEWS EXPERIMENT — does external info create an edge?")
    log(f"  {n} days | train {len(train)} val {len(val)} holdout {len(hold)}")
    log("=" * 64)

    reg = classify(g)
    tags = tag_days(dates)
    grid = build_grid()
    log("\nBuilding context cache...")
    cache = build_ctx_cache(g)
    log("  done.\n")

    allset = set(dates)
    trend = {d for d in dates if reg[d]["regime"] == "trend"}
    chop  = {d for d in dates if reg[d]["regime"] == "chop"}
    nonews = {d for d in dates if not tags[d]}
    trend_nonews = trend & nonews

    filters = [
        ("baseline",        allset),
        ("trend_only",      trend),
        ("chop_only",       chop),
        ("skip_news",       nonews),
        ("trend_skipnews",  trend_nonews),
    ]

    for name, allowed in filters:
        log(f"\n--- FILTER: {name}  ({len(allowed)} allowed days) ---")
        pipeline(name, dates, train, val, hold, cache, grid, allowed)
        walk_forward(name, dates, cache, grid, allowed)

    log("\n" + "=" * 64)
    log("  Interpretation: a filter 'works' only if its holdout flips to GO")
    log("  AND walk-forward shows PF>1.3. Otherwise the info didn't add a")
    log("  robust edge (removing trades can flatter a sample by luck).")
    log("=" * 64)
    OUT.close()


if __name__ == "__main__":
    main()
