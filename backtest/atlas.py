"""
MARKET ATLAS — systematic pattern battery on ES (2024-06 .. 2026-06).

Mission (user, 2026-06-11): "fully analyze the market... every change, every point,
any trend... pick up patterns and learn." This module measures EVERY classic effect
we can test on our data, with honest statistics, so strategy-building starts from
measured structure instead of vibes.

Honesty rules:
  - Every effect gets: n, effect size, naive t-stat.
  - We run ~12 studies x many buckets => expect ~5% false positives at |t|>2.
    PROMOTION BAR: |t| >= 3 AND economically meaningful AND theoretically sensible.
  - Everything here is IN-SAMPLE exploration. Any strategy built from it still
    faces the full validation gauntlet (OOS, random benchmark, Reality Check).

Run: python -m backtest.atlas        (results interpreted in docs/MARKET_ATLAS.md)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.config import CACHE_DIR

RTH_START, RTH_END = 9 * 60 + 30, 16 * 60


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def header(s):
    print("\n" + "=" * 92)
    print(f"  {s}")
    print("=" * 92)


def load():
    df = load_cached_24h()
    j = Path(CACHE_DIR) / "ESM6_june_tmp.parquet"
    if j.exists():
        x = pd.read_parquet(j)
        x["timestamp"] = pd.to_datetime(x["timestamp"]).dt.tz_convert("America/New_York")
        df = pd.concat([df, x]).drop_duplicates("timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)
    ts = df["timestamp"]
    df["mins"] = ts.dt.hour * 60 + ts.dt.minute
    df["date"] = ts.dt.date
    return df


def daily_frame(df):
    """Per trading day: RTH OHLC, prior 16:00 close, overnight H/L, returns."""
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    g = rth.groupby("date")
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "vol": g["volume"].sum(),
    })
    d["range"] = d.high - d.low
    d["prev_close"] = d.close.shift(1)
    d["gap"] = d.open - d.prev_close                      # overnight move (RTH close->open)
    d["intraday"] = d.close - d.open                      # open->close
    d["ret_cc"] = d.close.diff()
    d["dow"] = pd.to_datetime(d.index.astype(str)).dayofweek
    return d.dropna()


# --------------------------------------------------------------------------- #
def study_overnight_vs_intraday(d):
    header("A. OVERNIGHT vs INTRADAY drift (the classic decomposition)")
    on, intra = d.gap, d.intraday
    print(f"  overnight (close->open): total {on.sum():+.1f} pts | mean {on.mean():+.3f}/day | t={tstat(on):+.2f} | n={len(on)}")
    print(f"  intraday  (open->close): total {intra.sum():+.1f} pts | mean {intra.mean():+.3f}/day | t={tstat(intra):+.2f}")
    print(f"  close-to-close          : total {d.ret_cc.sum():+.1f} pts")
    print("  -> If overnight carries the drift and intraday ~0, ALL long-bias day strategies")
    print("     are fighting for scraps; the drift is collected while we're flat.")


def study_time_of_day(df):
    header("B. TIME-OF-DAY volatility & drift (30-min buckets, full 24h)")
    x = df.copy()
    x["ret"] = x.close.diff()
    x = x.dropna()
    x["bucket"] = (x.mins // 30) * 30
    g = x.groupby("bucket")["ret"]
    print(f"  {'bucket':>7} {'mean':>8} {'|mean|t':>8} {'std':>7} {'n':>7}")
    for b, s in g:
        if len(s) < 500:
            continue
        hh, mm = int(b) // 60, int(b) % 60
        print(f"  {hh:02d}:{mm:02d}  {s.mean():>8.3f} {tstat(s):>8.2f} {s.std():>7.2f} {len(s):>7}")
    print("  -> vol smile = where the action is; any |t|>3 bucket = candidate drift window.")


def study_gap_fill(d):
    header("C. GAP behavior (overnight move vs same-day fill)")
    d2 = d.copy()
    d2["gap_abs"] = d2.gap.abs()
    d2["filled"] = ((d2.gap > 0) & (d2.low <= d2.prev_close)) | \
                   ((d2.gap < 0) & (d2.high >= d2.prev_close))
    d2["with_gap"] = np.sign(d2.gap) * d2.intraday        # intraday continuation of gap
    for lo, hi, nm in [(0, 10, "small 0-10"), (10, 25, "mid 10-25"),
                       (25, 60, "big 25-60"), (60, 1e9, "huge >60")]:
        s = d2[(d2.gap_abs > lo) & (d2.gap_abs <= hi)]
        if len(s) < 10:
            continue
        print(f"  {nm:<12} n={len(s):>3}  P(fill same day)={s.filled.mean()*100:>3.0f}%  "
              f"gap-direction intraday follow: {s.with_gap.mean():+.2f} pts (t={tstat(s.with_gap):+.2f})")
    print("  -> classic: small gaps fill, huge gaps trend. Check which holds HERE.")


def study_momentum_horizons(df):
    header("D. MOMENTUM vs MEAN-REVERSION horizon scan (RTH, serial corr of k-bar returns)")
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    c = rth.close.to_numpy()
    dts = rth.date.to_numpy()
    print(f"  {'horizon':>9} {'corr':>8} {'t~corr*sqrt(n)':>15} {'n':>8}")
    for k, nm in [(1, "5min"), (3, "15min"), (6, "30min"), (12, "60min"), (24, "2h")]:
        r = c[k:] - c[:-k]
        same = dts[k:] == dts[:-k]                  # no overnight contamination
        r1, r2 = r[:-k][same[:-k] & same[k:]], r[k:][same[:-k] & same[k:]]
        if len(r1) < 100:
            continue
        cor = np.corrcoef(r1, r2)[0, 1]
        print(f"  {nm:>9} {cor:>8.4f} {cor*np.sqrt(len(r1)):>15.2f} {len(r1):>8}")
    print("  -> negative = mean reversion at that horizon; positive = momentum.")


def study_vol_clustering(d):
    header("E. VOLATILITY persistence (today's range vs tomorrow's)")
    r = d.range.to_numpy()
    cor1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    cor5 = np.corrcoef(r[:-5], r[5:])[0, 1]
    print(f"  corr(range_t, range_t+1) = {cor1:.3f}  (t≈{cor1*np.sqrt(len(r)-1):.1f})")
    print(f"  corr(range_t, range_t+5) = {cor5:.3f}")
    hi = r[:-1] > np.median(r); lo = ~hi
    print(f"  after HIGH-vol day: next range avg {r[1:][hi].mean():.1f} pts | after LOW-vol: {r[1:][lo].mean():.1f}")
    print("  -> the one effect everyone agrees is real. Quantified for OUR data.")


def study_first_hour(df, d):
    header("F. FIRST-HOUR range -> rest of day")
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    rows = []
    for date, day in rth.groupby("date"):
        am = day[day.mins < RTH_START + 60]
        pm = day[day.mins >= RTH_START + 60]
        if len(am) < 10 or len(pm) < 30:
            continue
        ah, al = am.high.max(), am.low.min()
        rows.append({
            "amr": ah - al,
            "broke_up": float(pm.high.max() > ah),
            "broke_dn": float(pm.low.min() < al),
            "both": float(pm.high.max() > ah and pm.low.min() < al),
            "pm_net": pm.close.iloc[-1] - pm.open.iloc[0],
        })
    f = pd.DataFrame(rows)
    print(f"  n={len(f)} days | P(PM breaks AM high)={f.broke_up.mean()*100:.0f}% | "
          f"P(breaks AM low)={f.broke_dn.mean()*100:.0f}% | P(breaks BOTH)={f.both.mean()*100:.0f}%")
    nr = f[f.amr < f.amr.quantile(0.3)]; wr = f[f.amr > f.amr.quantile(0.7)]
    print(f"  narrow AM range: P(both broken)={nr.both.mean()*100:.0f}%  | wide AM: {wr.both.mean()*100:.0f}%")
    print("  -> one-side break is near-certain (range expansion); BOTH broken = chop tax on breakouts.")


def study_pdh_pdl(df, d):
    header("G. PRIOR-DAY HIGH/LOW touch outcomes (the liquidity-level question)")
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    res = {"bounce": 0, "break": 0, "n": 0}
    fwd_after_touch = []
    pdh = d.high.shift(1); pdl = d.low.shift(1)
    for date, day in rth.groupby("date"):
        if date not in pdh.index or pd.isna(pdh.loc[date]):
            continue
        H = pdh.loc[date]
        c = day.close.to_numpy(); h = day.high.to_numpy()
        hit = np.nonzero(h >= H)[0]
        if len(hit) == 0 or hit[0] > len(c) - 13:
            continue
        i = hit[0]
        res["n"] += 1
        move = c[i + 12] - c[i]               # 1 hour after first touch
        fwd_after_touch.append(move)
        if move < -2: res["bounce"] += 1
        elif move > 2: res["break"] += 1
    fa = np.array(fwd_after_touch)
    print(f"  first touch of PRIOR-DAY HIGH: n={res['n']}  1h-later: mean {fa.mean():+.2f} pts (t={tstat(fa):+.2f})")
    print(f"  P(fade >2pts)={res['bounce']/res['n']*100:.0f}%  P(continue >2pts)={res['break']/res['n']*100:.0f}%")
    print("  -> tests 'liquidity level = reversal' directly, on first touch, 1h horizon.")


def study_streaks(d):
    header("H. DAILY STREAKS (does yesterday predict today?)")
    up = d.ret_cc > 0
    nxt = d.ret_cc.shift(-1)
    print(f"  after UP day  : next-day mean {nxt[up].mean():+.2f} pts (t={tstat(nxt[up]):+.2f}, n={up.sum()})")
    print(f"  after DOWN day: next-day mean {nxt[~up].mean():+.2f} pts (t={tstat(nxt[~up]):+.2f}, n={(~up).sum()})")
    s2 = up & up.shift(1).fillna(False); d2 = ~up & ~up.shift(1).fillna(True)
    print(f"  after 2 UP    : {nxt[s2].mean():+.2f} (n={s2.sum()})   after 2 DOWN: {nxt[d2].mean():+.2f} (n={d2.sum()})")


def study_dow(d):
    header("I. DAY-OF-WEEK (expect noise — shown for honesty)")
    for i, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        s = d[d.dow == i].ret_cc
        print(f"  {nm}: mean {s.mean():+.2f} pts  t={tstat(s):+.2f}  n={len(s)}")


def study_range_pos_close(d):
    header("J. CLOSE LOCATION vs NEXT DAY (close near high/low -> follow-through?)")
    pos = (d.close - d.low) / d.range.replace(0, np.nan)
    nxt = d.ret_cc.shift(-1)
    hi = pos > 0.8; lo = pos < 0.2
    print(f"  close in top 20% of range : next-day {nxt[hi].mean():+.2f} pts (t={tstat(nxt[hi]):+.2f}, n={hi.sum()})")
    print(f"  close in bottom 20%       : next-day {nxt[lo].mean():+.2f} pts (t={tstat(nxt[lo]):+.2f}, n={lo.sum()})")


def study_last_hour(df):
    header("K. LAST-HOUR behavior (15:00-16:00) conditioned on day so far")
    rth = df[(df.mins >= RTH_START) & (df.mins < RTH_END)]
    rows = []
    for date, day in rth.groupby("date"):
        upto = day[day.mins < 15 * 60]
        last = day[day.mins >= 15 * 60]
        if len(upto) < 30 or len(last) < 6:
            continue
        day_net = upto.close.iloc[-1] - upto.open.iloc[0]
        rows.append({"day_net": day_net, "lh": last.close.iloc[-1] - last.open.iloc[0]})
    f = pd.DataFrame(rows)
    up = f[f.day_net > 10]; dn = f[f.day_net < -10]
    print(f"  day up >10pts by 15:00  : last hour {up['lh'].mean():+.2f} pts (t={tstat(up['lh']):+.2f}, n={len(up)})")
    print(f"  day down >10pts by 15:00: last hour {dn['lh'].mean():+.2f} pts (t={tstat(dn['lh']):+.2f}, n={len(dn)})")
    print("  -> tests momentum-into-close (MOC effect) vs late-day mean reversion.")


def main():
    df = load()
    d = daily_frame(df)
    print(f"data: {len(df)} 5-min bars, {len(d)} trading days "
          f"({d.index[0]} .. {d.index[-1]})")
    study_overnight_vs_intraday(d)
    study_time_of_day(df)
    study_gap_fill(d)
    study_momentum_horizons(df)
    study_vol_clustering(d)
    study_first_hour(df, d)
    study_pdh_pdl(df, d)
    study_streaks(d)
    study_dow(d)
    study_range_pos_close(d)
    study_last_hour(df)
    print("\nPROMOTION BAR: |t|>=3 AND economically meaningful AND makes sense. Everything")
    print("else is noise until proven otherwise. Next: docs/MARKET_ATLAS.md interprets this.")


if __name__ == "__main__":
    main()
