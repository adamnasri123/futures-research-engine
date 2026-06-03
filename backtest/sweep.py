"""
Broad parameter sweep with sacred holdout + walk-forward — the honest way.

Phases:
  1. Build a per-day context cache for every (exec_tf, trend_tf) pair (causal).
  2. TRAIN (first 50%): run the full grid, rank by net profit factor.
  3. VALIDATION (next 25%): re-check the top combos; keep those that hold up
     (val PF > 1.1 AND val PF >= 0.6 * train PF — a parameter must generalize).
  4. HOLDOUT (final 25%, never used to choose anything): test the few survivors
     ONCE, with a direction-matched random benchmark + 2-tick stress + MLL check.
  5. WALK-FORWARD: independently, re-optimize the grid on a rolling window and
     trade the next unseen window; stitch the out-of-sample results. This asks
     "does optimizing on the past actually predict the future?"

Run: python -m backtest.sweep
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.modular import (
    DayCtx, Params, run_ctx, random_outcomes_ctx, ENTRY_METHODS, EXIT_METHODS,
)
from backtest.config import (
    SWEEP_TRAIN_FRAC, SWEEP_VAL_FRAC, SWEEP_MIN_TRADES, SLIPPAGE_TICKS,
    SLIPPAGE_STRESS_TICKS, TRAILING_MLL,
)

TF_PAIRS = [(5, 5), (5, 20), (15, 15), (15, 60), (30, 30), (30, 60)]
STOPS    = [1.5, 2.5]

OUT = open("backtest/sweep_results.txt", "w")
def log(msg=""):
    print(msg)
    OUT.write(msg + "\n")
    OUT.flush()


def build_grid():
    grid = []
    for (etf, ttf) in TF_PAIRS:
        for entry in ENTRY_METHODS:
            for exit_ in EXIT_METHODS:
                for stop in STOPS:
                    grid.append(Params(etf, ttf, entry, exit_, stop))
    return grid


def stats(net_list):
    n = len(net_list)
    if n == 0:
        return None
    arr = np.asarray(net_list, dtype=float)
    wins = arr[arr > 0]; losses = arr[arr <= 0]
    gl = -losses.sum()
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    return {"n": n, "pf": pf, "net": float(arr.sum()),
            "exp": float(arr.mean()), "wr": len(wins) / n}


def build_ctx_cache(day_groups):
    """ctx depends only on (date, exec_tf, trend_tf) — build once, reuse everywhere."""
    cache = {}
    pairs = sorted(set(TF_PAIRS))
    for (etf, ttf) in pairs:
        probe = Params(etf, ttf, "breakout", "swing", 2.0)  # entry/exit/stop irrelevant to ctx
        for date, day5m in day_groups:
            cache[(date, etf, ttf)] = DayCtx(day5m, probe)
    return cache


def run_combo(dates, cache, p, slippage=SLIPPAGE_TICKS):
    nets = []
    for date in dates:
        ctx = cache[(date, p.exec_tf, p.trend_tf)]
        t = run_ctx(ctx, date, p)
        if t is not None:
            nets.append(apply_costs(t.gross_pnl, slippage))
    return nets


def matched_benchmark(dates, cache, p, runs=500):
    per_day = []
    for date in dates:
        ctx = cache[(date, p.exec_tf, p.trend_tf)]
        outs = random_outcomes_ctx(ctx, p)
        if outs:
            per_day.append(outs)
    if not per_day:
        return None
    totals = np.empty(runs)
    for r in range(runs):
        totals[r] = sum(random.choice(o) for o in per_day)
    totals.sort()
    return {"mean": float(totals.mean()),
            "p95": float(totals[int(runs * 0.95)]),
            "pos": float((totals > 0).mean())}


def main():
    random.seed(42)
    day_groups = group_by_day(load_cached())
    dates_all = [d for d, _ in day_groups]
    n = len(dates_all)
    i_tr = int(n * SWEEP_TRAIN_FRAC)
    i_va = int(n * (SWEEP_TRAIN_FRAC + SWEEP_VAL_FRAC))
    train, val, hold = dates_all[:i_tr], dates_all[i_tr:i_va], dates_all[i_va:]

    log("=" * 64)
    log("  BROAD SWEEP — sacred holdout + walk-forward")
    log(f"  {n} days  ->  train {len(train)} | val {len(val)} | holdout {len(hold)}")
    log("=" * 64)

    grid = build_grid()
    log(f"\nGrid: {len(grid)} combinations "
        f"({len(TF_PAIRS)} TF pairs x {len(ENTRY_METHODS)} entries x "
        f"{len(EXIT_METHODS)} exits x {len(STOPS)} stops)")

    log("\nBuilding per-day context cache (causal)...")
    cache = build_ctx_cache(day_groups)
    log("  done.")

    # --- TRAIN ---
    log("\n[1] TRAIN — ranking full grid by net profit factor...")
    train_rows = []
    for p in grid:
        s = run_combo(train, cache, p)
        st = stats(s)
        if st and st["n"] >= SWEEP_MIN_TRADES:
            train_rows.append((p, st))
    train_rows.sort(key=lambda r: r[1]["pf"], reverse=True)

    log(f"  {len(train_rows)} combos with >= {SWEEP_MIN_TRADES} trades. Top 12 on TRAIN:")
    log(f"    {'combo':<42} {'n':>4} {'PF':>5} {'net$':>8} {'win%':>5}")
    for p, st in train_rows[:12]:
        log(f"    {p.label():<42} {st['n']:>4} {st['pf']:>5.2f} {st['net']:>8.0f} {st['wr']*100:>5.0f}")

    # --- VALIDATION ---
    log("\n[2] VALIDATION — keep combos that generalize (val PF>1.1 & >=0.6x train PF)...")
    top_train = [r for r in train_rows if r[1]["pf"] > 1.1][:30]
    candidates = []
    for p, tr_st in top_train:
        va_st = stats(run_combo(val, cache, p))
        if va_st and va_st["n"] >= SWEEP_MIN_TRADES * 0.5 and va_st["pf"] > 1.1 \
           and va_st["pf"] >= 0.6 * tr_st["pf"]:
            candidates.append((p, tr_st, va_st))
    candidates.sort(key=lambda r: r[2]["pf"], reverse=True)

    if not candidates:
        log("  NONE survived validation. No combo generalized from train to val.")
        log("\n" + "=" * 64)
        log("  RESULT: NO-GO — the broad search found no strategy that holds up")
        log("  out-of-sample. (Expected: most do not.)")
        log("=" * 64)
    else:
        log(f"  {len(candidates)} candidates survived. Showing up to 8:")
        log(f"    {'combo':<42} {'trPF':>5} {'vaPF':>5} {'vaNet$':>8}")
        for p, tr, va in candidates[:8]:
            log(f"    {p.label():<42} {tr['pf']:>5.2f} {va['pf']:>5.2f} {va['net']:>8.0f}")

        # --- HOLDOUT (sacred) ---
        log("\n[3] HOLDOUT (sacred, used once) — survivors vs random + stress + MLL...")
        survivors = candidates[:6]
        any_go = False
        for p, tr, va in survivors:
            nets = run_combo(hold, cache, p)
            st = stats(nets)
            if st is None:
                continue
            stress = stats(run_combo(hold, cache, p, slippage=SLIPPAGE_STRESS_TICKS))
            bench = matched_benchmark(hold, cache, p, runs=500)
            breach, _ = simulate_mll(nets)
            beats = bench is not None and st["net"] > bench["p95"]
            gates = {
                "n>=30":           st["n"] >= 30,
                "PF>1.3":          st["pf"] > 1.3,
                "beats random":    beats,
                "no MLL breach":   not breach,
                "stress PF>1.0":   stress is not None and stress["pf"] > 1.0,
            }
            ok = all(gates.values())
            any_go = any_go or ok
            log(f"\n  {p.label()}")
            log(f"    holdout: n={st['n']} PF={st['pf']:.2f} net=${st['net']:.0f} "
                f"win={st['wr']*100:.0f}%  stressPF={stress['pf']:.2f}" if stress else "")
            if bench:
                log(f"    random: mean=${bench['mean']:.0f} p95=${bench['p95']:.0f} "
                    f"pos={bench['pos']*100:.0f}%  -> beats95={'YES' if beats else 'NO'}")
            log("    gates: " + "  ".join(f"{k}={'Y' if v else 'N'}" for k, v in gates.items()))
            log(f"    >>> {'GO' if ok else 'NO-GO'}")

        log("\n" + "=" * 64)
        log(f"  HOLDOUT RESULT: {'AT LEAST ONE GO — see above' if any_go else 'NO-GO (no survivor cleared all gates)'}")
        log("=" * 64)

    # --- WALK-FORWARD (independent honesty check) ---
    log("\n[4] WALK-FORWARD — re-optimize on rolling window, trade next unseen window...")
    wf_train, wf_test, wf_step = 150, 50, 50
    wf_nets = []
    start = 0
    picks = []
    while start + wf_train + wf_test <= n:
        tr_dates = dates_all[start:start + wf_train]
        te_dates = dates_all[start + wf_train:start + wf_train + wf_test]
        best, best_pf = None, -1
        for p in grid:
            st = stats(run_combo(tr_dates, cache, p))
            if st and st["n"] >= SWEEP_MIN_TRADES and st["pf"] > best_pf:
                best, best_pf = p, st["pf"]
        if best is not None:
            te_nets = run_combo(te_dates, cache, best)
            wf_nets.extend(te_nets)
            picks.append(best.label())
        start += wf_step

    wf = stats(wf_nets)
    if wf:
        breach, _ = simulate_mll(wf_nets)
        log(f"  stitched OOS: n={wf['n']} PF={wf['pf']:.2f} net=${wf['net']:.0f} "
            f"exp=${wf['exp']:.2f} win={wf['wr']*100:.0f}%  MLLbreach={breach}")
        log(f"  windows: {len(picks)} | distinct picks: {len(set(picks))}")
        log(f"  verdict: {'WF edge present (PF>1.3)' if wf['pf'] > 1.3 else 'NO WF edge (PF<=1.3)'}")
    else:
        log("  walk-forward produced no trades.")

    log("\nDONE. Full results saved to backtest/sweep_results.txt")
    OUT.close()


if __name__ == "__main__":
    main()
