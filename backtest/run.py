"""
Honest backtest runner — single entry point for every strategy.

Usage:
    python -m backtest.run orb       # 5-minute ORB + VWAP
    python -m backtest.run trend     # wave-rider, long + short
    python -m backtest.run           # defaults to orb

Each run does: full / in-sample / out-of-sample / 2-tick stress, plus a
strategy-appropriate random-entry benchmark, then a GO / NO-GO verdict against
the same six gates. A real edge must clear ALL of them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.data import load_cached, group_by_day
from backtest.strategy import run_day
from backtest.strategy_trend import run_day_trend_ls, run_day_trend_immediate
from backtest.benchmark import (
    run_benchmark, direction_matched_benchmark, direction_matched_benchmark_ride,
)
from backtest.config import (
    IN_SAMPLE_FRACTION, SLIPPAGE_TICKS, SLIPPAGE_STRESS_TICKS, BENCHMARK_RUNS,
)

STRATEGIES = {
    "orb": {
        "fn":    run_day,
        "bench": run_benchmark,
        "title": "5-Minute ORB + VWAP",
    },
    "trend": {
        "fn":    run_day_trend_ls,
        "bench": direction_matched_benchmark,
        "title": "Wave-Rider (structure trend-ride, long + short)",
    },
    "ride": {
        "fn":    run_day_trend_immediate,
        "bench": direction_matched_benchmark_ride,
        "title": "Trend-Ride v2 (enter sooner, wide stop, ride to close)",
    },
}


def split_in_out(results):
    days = sorted(results["date"].unique())
    cut  = int(len(days) * IN_SAMPLE_FRACTION)
    return (results[results["date"].isin(set(days[:cut]))].copy(),
            results[results["date"].isin(set(days[cut:]))].copy())


def _report_sides(traded):
    """Optional extra reporting: side split + exit-reason breakdown."""
    if traded.empty:
        return
    if traded["side"].nunique() > 1 or (traded["side"] == "short").any():
        longs  = traded[traded["side"] == "long"]
        shorts = traded[traded["side"] == "short"]
        print(f"\n  Side split: {len(longs)} longs (net ${longs['net_pnl'].sum():.0f}), "
              f"{len(shorts)} shorts (net ${shorts['net_pnl'].sum():.0f})")
    if "exit_reason" in traded.columns and traded["exit_reason"].any():
        print("  Exit reasons:")
        for reason, grp in traded.groupby("exit_reason"):
            if reason:
                print(f"    {reason:16s}: {len(grp):3d} trades, "
                      f"avg ${grp['net_pnl'].mean():7.2f}, total ${grp['net_pnl'].sum():8.2f}")


def run_strategy(name: str):
    cfg = STRATEGIES[name]
    print("\n" + "=" * 55)
    print(f"  HONEST BACKTEST — {cfg['title']}")
    print(f"  ES 5-min bars (Massive.com), full RTH")
    print("=" * 55)

    groups = group_by_day(load_cached())
    print(f"\nLoaded {len(groups)} trading days.")

    print("\n[Full / In-sample / Out-of-sample]")
    res     = run_backtest(day_groups=groups, strategy_fn=cfg["fn"], slippage_ticks=SLIPPAGE_TICKS)
    m_full  = compute_metrics(res, "FULL")
    ins, outs = split_in_out(res)
    m_in    = compute_metrics(ins,  "IN-SAMPLE (70%)")
    m_out   = compute_metrics(outs, "OUT-OF-SAMPLE (30%)")

    print(f"\n[Stress test: {SLIPPAGE_STRESS_TICKS}-tick slippage]")
    stress  = run_backtest(day_groups=groups, strategy_fn=cfg["fn"], slippage_ticks=SLIPPAGE_STRESS_TICKS)
    _, stress_out = split_in_out(stress)
    m_stress = compute_metrics(stress_out, f"STRESS ({SLIPPAGE_STRESS_TICKS}-tick) OOS")

    _report_sides(res[~res["skipped"]])

    print(f"\n[Random-entry benchmark: {BENCHMARK_RUNS} simulations]")
    bench = cfg["bench"](n_runs=BENCHMARK_RUNS, day_groups=groups)
    real_total = m_full.get("total_net_pnl", 0)
    beats = real_total > bench["pct_95"]
    print(f"  random: mean ${bench['mean']:.0f} | median ${bench['median']:.0f} "
          f"| 95th ${bench['pct_95']:.0f} | %pos {bench['pct_positive']:.1%}")
    print(f"  REAL strategy: ${real_total:.0f}")
    print(f"  Beats random 95th pct: {'YES' if beats else 'NO'}")

    # --- six honest gates ---
    out_pf = m_out.get("profit_factor", 0)
    in_pf  = m_in.get("profit_factor", 0)
    str_pf = m_stress.get("profit_factor", 0)
    n_all  = m_in.get("n_trades", 0) + m_out.get("n_trades", 0)

    checks = {
        f"100+ trades (got {n_all})":                                n_all >= 100,
        f"Out-of-sample PF > 1.3 (got {out_pf:.2f})":                out_pf > 1.3,
        f"OOS not much worse than IS ({in_pf:.2f} -> {out_pf:.2f})": out_pf >= in_pf * 0.6,
        f"Beats random entry 95th pct":                             beats,
        f"No MLL breach at 1 contract":                             not m_out.get("mll_breach", True),
        f"Stress-test PF > 1.0 (got {str_pf:.2f})":                 str_pf > 1.0,
    }

    print("\n" + "=" * 55)
    print("  GO / NO-GO CHECKLIST")
    print("=" * 55)
    all_pass = True
    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 55)
    if all_pass:
        print("  VERDICT: GO — run 1 MES contract on the Combine, monitor closely.")
    else:
        print("  VERDICT: NO-GO — do not fund a live account on this.")
    print("=" * 55 + "\n")


def main():
    name = sys.argv[1].lower() if len(sys.argv) > 1 else "orb"
    if name not in STRATEGIES:
        print(f"Unknown strategy '{name}'. Choose from: {', '.join(STRATEGIES)}")
        sys.exit(1)
    run_strategy(name)


if __name__ == "__main__":
    main()
