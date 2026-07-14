"""
MULTI-DAY MEAN REVERSION on the S&P — testing the Baltussen/van Bekkum/Da (JFE 2019)
mechanism: index-product growth flipped serial dependence NEGATIVE after ~2000.

Data: SPX daily 1980-2026 (Yahoo, cached SPX_daily_yahoo.parquet). Signals on the
index; execution would be MES.

PART 1 — mechanism by era: lag-1 autocorr, past-5d->next-1d corr, and 3d->3d corr
per 5-year window. Key question: still negative 2016-2026?

PART 2 — TAIL strategy (the only cost-viable form): z = 3-day return / rolling 63d
sigma of 3-day returns (causal). Enter LONG at close when z < -k; exit after M days
or when z >= 0. SHORT symmetric (reported separately; shorts fight drift).
Params k in {1.5, 2.0, 2.5} x M in {3, 5}: chosen on TRAIN = 2000-2015,
OOS = 2016-2026 touched once. 1980-1999 shown as the wrong-era control.

Costs (TopStep-realistic): entry+exit ~1.2bp total PLUS 1.2bp per overnight held
(the eval forces flat 16:10->18:00, so a multi-day hold = flatten & re-enter daily).
Also: $-terms sim for 2016-2026 at 1 and 3 MES micros with DAILY mark-to-market
equity for the $2,000 trailing-MLL check (EOD marks, like TopStep).

Run: python -m backtest.meanrev
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.config import CACHE_DIR

RT_COST_BP = 1.2          # round-turn, % basis points (0.62 ES pts at ~6000 = ~1.03bp)
OVERNIGHT_BP = 1.2        # per held overnight (forced flatten + re-enter)
POINT_VALUE = 5.0


def load_spx():
    df = pd.read_parquet(Path(CACHE_DIR) / "SPX_daily_yahoo.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret1"] = df.close.pct_change()
    return df.dropna().reset_index(drop=True)


def era_table(df):
    print("=" * 88)
    print("  PART 1 — SERIAL DEPENDENCE BY ERA (the mechanism check)")
    print("=" * 88)
    print(f"  {'era':<12}{'n':>6}{'lag1 AC':>9}{'t':>7}{'5d->1d':>9}{'t':>7}{'3d->3d':>9}{'t':>7}")
    r = df.ret1.to_numpy()
    yrs = df.date.dt.year.to_numpy()
    for y0 in range(1980, 2026, 5):
        m = (yrs >= y0) & (yrs < y0 + 5)
        if m.sum() < 300:
            # final partial era
            m = (yrs >= y0)
            if m.sum() < 200:
                continue
        x = r[m]
        n = len(x)
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        p5 = pd.Series(x).rolling(5).mean().shift(1).to_numpy()
        ok = ~np.isnan(p5)
        c51 = np.corrcoef(p5[ok], x[ok])[0, 1]
        r3 = pd.Series(x).rolling(3).sum().to_numpy()
        a, b = r3[3:-3:3], r3[6::3]          # non-overlapping 3d blocks
        k = min(len(a), len(b))
        c33 = np.corrcoef(a[:k], b[:k])[0, 1]
        lbl = f"{y0}-{min(y0+4, 2026)}"
        print(f"  {lbl:<12}{n:>6}{ac1:>9.3f}{ac1*np.sqrt(n):>7.1f}"
              f"{c51:>9.3f}{c51*np.sqrt(ok.sum()):>7.1f}{c33:>9.3f}{c33*np.sqrt(k):>7.1f}")
    print("  -> negative numbers post-2000 = the paper's regime. CRITICAL: 2016+ rows.")


def tail_trades(df, k, M, side="long"):
    """Enter at close when z crosses the tail; exit after M days or z>=0.
    Returns list of dicts with pct P&L net of costs and holding days."""
    c = df.close.to_numpy()
    r3 = df.close.pct_change(3).to_numpy()
    sig = pd.Series(r3).rolling(63).std().shift(1).to_numpy()
    z = r3 / sig
    yrs = df.date.dt.year.to_numpy()
    n = len(df)
    trades = []
    i = 63
    while i < n - 1:
        hit = (z[i] < -k) if side == "long" else (z[i] > k)
        if not np.isnan(z[i]) and hit:
            j_exit = min(i + M, n - 1)
            for j in range(i + 1, j_exit + 1):
                if (side == "long" and z[j] >= 0) or (side == "short" and z[j] <= 0):
                    j_exit = j
                    break
            gross = (c[j_exit] / c[i] - 1) * (1 if side == "long" else -1)
            held = j_exit - i
            cost = (RT_COST_BP + OVERNIGHT_BP * max(0, held - 1)) / 10000
            trades.append({"year": int(yrs[i]), "entry_i": i, "exit_i": j_exit,
                           "pct": gross - cost, "held": held})
            i = j_exit + 1
        else:
            i += 1
    return trades


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def seg_stats(trades, y0, y1, label):
    s = [t["pct"] for t in trades if y0 <= t["year"] <= y1]
    if not s:
        return f"  {label:<26} (no trades)", None
    a = np.array(s)
    return (f"  {label:<26}{len(a):>5}{a.mean()*100:>9.3f}{a.sum()*100:>9.1f}"
            f"{(a>0).mean()*100:>6.0f}{tstat(a):>7.2f}"), a


def dollar_sim(df, trades, micros, y0=2016, y1=2026):
    """Daily mark-to-market $ equity for the OOS window at N micros; MLL check."""
    c = df.close.to_numpy()
    yrs = df.date.dt.year.to_numpy()
    pos_by_day = np.zeros(len(df))          # +1/-1 while a trade is on
    for t in trades:
        if y0 <= t["year"] <= y1:
            pos_by_day[t["entry_i"]:t["exit_i"]] = 1  # long variant only here
    eq = 0.0; peak = 0.0; mll = -2000.0; breach_day = None
    day_pnl = []
    for i in range(1, len(df)):
        if not (y0 <= yrs[i] <= y1):
            continue
        pnl = pos_by_day[i - 1] * (c[i] - c[i - 1]) * POINT_VALUE * micros
        # subtract churn costs on holding days
        if pos_by_day[i - 1]:
            pnl -= OVERNIGHT_BP / 10000 * c[i] * POINT_VALUE * micros
        eq += pnl
        day_pnl.append(pnl)
        if eq > peak:
            peak, mll = eq, eq - 2000.0
        if eq < mll and breach_day is None:
            breach_day = i
    a = np.array(day_pnl)
    dd = (np.maximum.accumulate(np.cumsum(a)) - np.cumsum(a)).max() if len(a) else 0
    return a.sum(), dd, breach_day is not None


def main():
    df = load_spx()
    print(f"SPX daily: {len(df)} days {df.date.iloc[0].date()} .. {df.date.iloc[-1].date()}\n")
    era_table(df)

    print("\n" + "=" * 88)
    print("  PART 2 — TAIL MEAN-REVERSION STRATEGY (LONG side)")
    print("=" * 88)
    hdr = f"  {'segment':<26}{'n':>5}{'avg%':>9}{'tot%':>9}{'win%':>6}{'t':>7}"

    # param selection on TRAIN only
    best = None
    print("\n  TRAIN 2000-2015 grid (choose best by t-stat, then OOS ONCE):")
    print(hdr)
    for k in (1.5, 2.0, 2.5):
        for M in (3, 5):
            tr = tail_trades(df, k, M, "long")
            line, a = seg_stats(tr, 2000, 2015, f"k={k} M={M}")
            print(line)
            if a is not None and len(a) >= 30:
                t = tstat(a)
                if best is None or t > best[0]:
                    best = (t, k, M, tr)

    t, k, M, tr = best
    print(f"\n  SELECTED on train: k={k}, M={M} (t={t:.2f}). Now the honest reads:")
    print(hdr)
    for y0, y1, lbl in [(1980, 1999, "1980-1999 (control era)"),
                        (2000, 2015, "2000-2015 (train)"),
                        (2016, 2026, "2016-2026 (OOS)"),
                        (2021, 2026, "2021-2026 (recent)")]:
        print(seg_stats(tr, y0, y1, lbl)[0])

    print("\n  SHORT side, same params (fights drift — shown for symmetry honesty):")
    trs = tail_trades(df, k, M, "short")
    print(hdr)
    for y0, y1, lbl in [(2000, 2015, "2000-2015"), (2016, 2026, "2016-2026 (OOS)")]:
        print(seg_stats(trs, y0, y1, lbl)[0])

    # random-entry control for the OOS long side: random entry days, same M-day exit
    rng = np.random.default_rng(5)
    c = df.close.to_numpy(); yrs = df.date.dt.year.to_numpy()
    oos_idx = np.nonzero((yrs >= 2016) & (yrs <= 2026))[0]
    n_oos = len([x for x in tr if 2016 <= x["year"] <= 2026])
    totals = []
    for _ in range(1000):
        picks = rng.choice(oos_idx[:-M-1], size=max(n_oos, 1), replace=False)
        tot = 0.0
        for i in picks:
            j = min(i + M, len(c) - 1)
            tot += (c[j]/c[i] - 1) - (RT_COST_BP + OVERNIGHT_BP*(M-1))/10000
        totals.append(tot * 100)
    print(f"\n  random-LONG control OOS (same n={n_oos}, hold {M}d, 1000 runs): "
          f"mean {np.mean(totals):.1f}%  p95 {np.percentile(totals, 95):.1f}%")
    oos_tot = sum(x['pct'] for x in tr if 2016 <= x['year'] <= 2026) * 100
    print(f"  strategy OOS total: {oos_tot:.1f}%  -> beats p95? {oos_tot > np.percentile(totals, 95)}")

    # $-reality at 1 and 3 micros, OOS decade, EOD-marked MLL
    print("\n  $-SIM 2016-2026 (EOD-marked equity, incl. daily churn):")
    for m in (1, 3):
        tot, dd, breach = dollar_sim(df, tr, m)
        print(f"    {m} micro(s): net ${tot:,.0f} | maxDD ${dd:,.0f} | "
              f"$2,000 trailing-MLL breach: {'YES' if breach else 'no'}")

    print("\n  NOTE: entries/exits at daily closes; MES only exists since 2019 — earlier")
    print("  years are signal validation, not literal P&L. Gates: OOS positive with t>=2,")
    print("  beats random p95, survives MLL at chosen size, THEN paper-forward.")


if __name__ == "__main__":
    main()
