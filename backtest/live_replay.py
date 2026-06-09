"""
LIVE-FAITHFUL REPLAY — backtests the strategy the bot ACTUALLY runs live.

Why this exists (2026-06-09 audit, see logs/RESEARCH_LOG.md): every prior "replay the
live strategy" test (sizing.py, entry_timing.py, goal_sizing.py) ran on the RTH-only
cache and warmed indicators up IN-DAY, so the earliest possible signal closed at
~10:45 ET. The live bot (autotrader.py) computes the same indicators on the LAST 8
HOURS of bars (incl. overnight Globex) and fires as early as 9:35 ET — which is when
5 of its first 6 real trades happened. The old replays therefore tested a different
strategy. This module replicates the live path exactly:

  - 24h data; signal evaluated on a 96-bar (8h) window ending at each completed bar,
    exactly mirroring autotrader.recent_5m + signal() + bracket_ticks().
  - Entry window by WALL CLOCK at decision time (bar CLOSE): 9:35 <= t <= 12:00
    (preflight checks now_min, not bar window_start).
  - Stop = round-to-tick(2.5 * ATR14 on the window); target = 2x stop ticks.
  - Entry fill = next bar OPEN (live sends market seconds after the bar completes).
  - Force-flat fill = OPEN of the first bar with window_start >= 15:55 (live closes
    at market at ~15:55:19). Old replays used that bar's CLOSE (16:00) — 5 min late.
  - Same-bar stop/target ambiguity resolved stop-first (conservative, as before).
  - One trade per day. Costs = backtest.costs.apply_costs (same as all other tests).

Modes:
  python -m backtest.live_replay parity   # June 1-9 2026 vs the real live trades
  python -m backtest.live_replay full     # full history, filters, sizes, benchmark
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.regime import _adx, _choppiness, ADX_TREND, ADX_CHOP, CHOP_TREND, CHOP_CHOP
from backtest.news import FOMC_DAYS, nfp_days_for_range
from backtest.config import POINT_VALUE, TICK_SIZE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION

# Live params (live_config.py)
EMA_TREND, BREAKOUT_N, ATR_PERIOD, STOP_ATR = 10, 6, 14, 2.5
WINDOW_BARS = 96                      # 8h of 5-min bars = autotrader's fetch window
DEC_START, DEC_END = 9*60+35, 12*60   # decision-time window (wall clock, bar close)
FLAT_MIN = 15*60+55

JUNE_TMP = Path("backtest/cache/ESM6_june_tmp.parquet")


def _ema_last2(c):
    """EMA over the window, seeded at window[0] exactly like autotrader.ema.
    Returns (ema[-1], ema[-2])."""
    k = 2.0 / (EMA_TREND + 1)
    e = c[0]
    prev = e
    for x in c[1:]:
        prev = e
        e = x * k + e * (1 - k)
    return e, prev


def _atr_last(h, l, c):
    """Wilder ATR14 over the window, same as autotrader.atr; returns last value."""
    tr = np.empty(len(h)); tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = tr[:ATR_PERIOD].mean()
    for i in range(ATR_PERIOD, len(h)):
        a = (a * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    return a


def load_frame(include_june=True):
    df = load_cached_24h()
    if include_june and JUNE_TMP.exists():
        j = pd.read_parquet(JUNE_TMP)
        j["timestamp"] = pd.to_datetime(j["timestamp"]).dt.tz_convert("America/New_York")
        df = pd.concat([df, j], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def replay(df, allowed_dates=None, dec_start=DEC_START, target_first=False):
    """Run the live-faithful replay. allowed_dates: set of date-strings to trade
    (day filter), or None = all days. dec_start: earliest decision time (minute of
    day) — set to 645 (10:45) to mimic the old RTH replays' entry population.
    target_first: resolve same-bar stop/target ambiguity optimistically (upper
    bound) instead of the default conservative stop-first. Returns trade dicts."""
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    ts = df["timestamp"]
    mins = (ts.dt.hour*60 + ts.dt.minute).to_numpy()
    dates = ts.dt.date.astype(str).to_numpy()

    trades = []
    n = len(df)
    # group indices by date once
    idx_by_date = {}
    for i, d in enumerate(dates):
        idx_by_date.setdefault(d, []).append(i)

    for d in sorted(idx_by_date):
        if allowed_dates is not None and d not in allowed_dates:
            continue
        day_idx = idx_by_date[d]
        for i in day_idx:
            dec_min = mins[i] + 5                      # decision happens at bar CLOSE
            if dec_min < dec_start or dec_min > DEC_END:
                continue
            if i < WINDOW_BARS or i + 1 >= n or dates[i+1] != d:
                continue
            w = slice(i - WINDOW_BARS + 1, i + 1)      # last 96 completed bars
            cw = c[w]
            e_now, e_prev = _ema_last2(cw)
            up = cw[-1] > e_now and e_now > e_prev
            dn = cw[-1] < e_now and e_now < e_prev
            sig = None
            hw = h[w]; lw = l[w]
            if up and cw[-1] > hw[-BREAKOUT_N-1:-1].max():
                sig = "long"
            elif dn and cw[-1] < lw[-BREAKOUT_N-1:-1].min():
                sig = "short"
            if sig is None:
                continue

            a = _atr_last(hw, lw, cw)
            stop_ticks = max(1, int(round(STOP_ATR * a / TICK_SIZE)))
            stop_pts = stop_ticks * TICK_SIZE
            tgt_pts = 2 * stop_pts
            entry = o[i+1]
            if sig == "long":
                stop_px, tgt_px = entry - stop_pts, entry + tgt_pts
            else:
                stop_px, tgt_px = entry + stop_pts, entry - tgt_pts

            outcome, exit_px = "session_end", c[day_idx[-1]]
            for j in range(i + 1, day_idx[-1] + 1):
                if dates[j] != d:
                    break
                if mins[j] >= FLAT_MIN:
                    outcome, exit_px = "time", o[j]; break
                if sig == "long":
                    hit_s, hit_t = l[j] <= stop_px, h[j] >= tgt_px
                else:
                    hit_s, hit_t = h[j] >= stop_px, l[j] <= tgt_px
                if target_first and hit_t:
                    outcome, exit_px = "target", tgt_px; break
                if hit_s:
                    outcome, exit_px = "stop", stop_px; break
                if hit_t:
                    outcome, exit_px = "target", tgt_px; break

            pts = (exit_px - entry) if sig == "long" else (entry - exit_px)
            trades.append({
                "date": d, "side": sig,
                "dec_time": f"{dec_min//60:02d}:{dec_min%60:02d}",
                "entry": entry, "stop_ticks": stop_ticks,
                "exit": exit_px, "outcome": outcome, "pts": pts,
                "net1": apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS),
            })
            break  # one trade/day
    return trades


# --------------------------------------------------------------------------- #
# Day filter — regime TREND + no FOMC/NFP, computed causally from trade-date
# daily bars (18:00 D-1 .. 17:00 D), roll-gap back-adjusted (closer to what the
# live bot sees than the old RTH-collapsed version).
# --------------------------------------------------------------------------- #
def trend_no_news_dates(df):
    ts = df["timestamp"]
    # back-adjust roll gaps measured at ticker transitions
    c_adj = df["close"].to_numpy().astype(float).copy()
    o_adj = df["open"].to_numpy().astype(float).copy()
    h_adj = df["high"].to_numpy().astype(float).copy()
    l_adj = df["low"].to_numpy().astype(float).copy()
    tk = df["ticker"].to_numpy()
    chg = np.nonzero(tk[1:] != tk[:-1])[0]
    for ix in chg:                       # subtract each gap from everything BEFORE it
        gap = c_adj[ix+1] - c_adj[ix]
        c_adj[:ix+1] += gap; o_adj[:ix+1] += gap
        h_adj[:ix+1] += gap; l_adj[:ix+1] += gap

    # trade-date = date of (ts + 6h): 18:00 D-1 evening session belongs to D
    trade_date = (ts + pd.Timedelta(hours=6)).dt.date.astype(str)
    tmp = pd.DataFrame({"td": trade_date, "h": h_adj, "l": l_adj, "c": c_adj})
    g = tmp.groupby("td", sort=True)
    td = list(g.groups.keys())
    H = g["h"].max().to_numpy(); L = g["l"].min().to_numpy(); C = g["c"].last().to_numpy()

    adx = _adx(H, L, C); chop = _choppiness(H, L, C)
    nfp = nfp_days_for_range(td[0], td[-1])
    out = set()
    for i, d in enumerate(td):
        if i < 1:
            continue
        a, ch = adx[i-1], chop[i-1]      # known at D's open
        if np.isnan(a) or np.isnan(ch):
            continue
        trend_votes = (a > ADX_TREND) + (ch < CHOP_TREND)
        chop_votes = (a < ADX_CHOP) + (ch > CHOP_CHOP)
        if trend_votes >= 1 and chop_votes == 0 and d not in FOMC_DAYS and d not in nfp:
            out.add(d)
    return out


# --------------------------------------------------------------------------- #
def stats(nets, label=""):
    a = np.asarray(nets, float)
    if len(a) == 0:
        return None
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    eq = np.cumsum(a); dd = float((eq - np.maximum.accumulate(eq)).min())
    breach, curve = simulate_mll(list(a))
    return {"label": label, "n": len(a), "net": float(a.sum()), "exp": float(a.mean()),
            "pf": pf, "wr": len(wins)/len(a), "maxdd": dd, "breach": breach,
            "breach_at": len(curve) if breach else None}


def print_stats_row(s):
    if s is None:
        print("  (no trades)"); return
    br = f"YES@#{s['breach_at']}" if s["breach"] else "no"
    print(f"  {s['label']:<26}{s['n']:>5}{s['net']:>9.0f}{s['exp']:>8.2f}{s['pf']:>6.2f}"
          f"{s['wr']*100:>6.0f}{s['maxdd']:>9.0f}{br:>10}")


HDR = f"  {'variant':<26}{'n':>5}{'net$':>9}{'exp$':>8}{'PF':>6}{'win%':>6}{'maxDD$':>9}{'MLL':>10}"


def run_parity():
    print("="*78)
    print("  PARITY TEST — live-faithful replay vs the 6 ACTUAL live trading days")
    print("="*78)
    df = load_frame(include_june=True)
    days = {"2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
            "2026-06-08", "2026-06-09"}          # logged TRADE days (06-05 = NFP aside)
    trades = replay(df, allowed_dates=days)
    live = {  # from logs/autotrader_*.log (signal px = last close at decision time)
        "2026-06-01": ("long", "11:35", 7598.5, 65, "target  (bot was DOWN 9:35-11:25 — bug fixes; expect replay to fire earlier)"),
        "2026-06-02": ("long", "09:35", 7602.5, 39, "target"),
        "2026-06-03": ("short", "09:45", 7592.75, 43, "time/manual? exit 7576.75"),
        "2026-06-04": ("long", "09:35", 7556.5, 57, "target"),
        "2026-06-08": ("short", "09:35", 7445.5, 73, "stop"),
        "2026-06-09": ("long", "09:40", 7466.0, 57, "stop (data ends ~10:55 today)"),
    }
    print(f"\n  {'date':<12}{'REPLAY':<38}{'LIVE':<40}")
    by_date = {t["date"]: t for t in trades}
    for d in sorted(live):
        lv = live[d]
        r = by_date.get(d)
        rtxt = (f"{r['side']:<6}{r['dec_time']} ent {r['entry']:.2f} stop {r['stop_ticks']}t "
                f"-> {r['outcome']}" if r else "NO TRADE")
        print(f"  {d:<12}{rtxt:<38}{lv[0]} {lv[1]} ~{lv[2]} stop {lv[3]}t -> {lv[4]}")
    print("\n  Match criteria: same side + decision time within 1 bar + stop ticks within")
    print("  ~10% (Massive vs TopStep feed differences). 06-01 expected to differ (downtime).")


def run_full():
    print("="*78)
    print("  LIVE-FAITHFUL REPLAY — full history (24h data, 8h window, 9:35 entries)")
    print("="*78)
    df = load_frame(include_june=True)
    trend_days = trend_no_news_dates(df)
    print(f"  bars={len(df)}  trend+no-news days={len(trend_days)}")

    for fname, allowed in [("ALL days", None), ("TREND+no-news (live)", trend_days)]:
        trades = replay(df, allowed_dates=allowed)
        nets1 = [t["net1"] for t in trades]
        cut = int(len(nets1) * IN_SAMPLE_FRACTION)
        early = sum(1 for t in trades if t["dec_time"] <= "10:45")
        print(f"\n### {fname}: {len(trades)} trades, {early} ({early/max(1,len(trades))*100:.0f}%) "
              f"decided at/before 10:45 ET (old replays sampled ~0% there)")
        print(HDR)
        print_stats_row(stats(nets1, "full, 1 micro"))
        print_stats_row(stats(nets1[cut:], "OOS, 1 micro"))
        for size in (2, 5):
            print_stats_row(stats([x*size for x in nets1[cut:]], f"OOS, {size} micros"))
        # outcome mix
        from collections import Counter
        cnt = Counter(t["outcome"] for t in trades)
        tot = sum(cnt.values())
        mix = "  ".join(f"{k}:{v} ({v/tot*100:.0f}%)" for k, v in cnt.most_common())
        print(f"  outcome mix: {mix}")
        med_stop = np.median([t["stop_ticks"] for t in trades]) if trades else 0
        print(f"  median stop: {med_stop:.0f} ticks ({med_stop*TICK_SIZE:.2f} pts)")

    # ---- random-entry benchmark on the live day-filter, same bracket logic ----
    print("\n### RANDOM-ENTRY BENCHMARK (same days/window/bracket, random bar+side, 500 runs)")
    bench(df, trend_days)


def bench(df, allowed, runs=500, seed=7):
    rng = np.random.default_rng(seed)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    ts = df["timestamp"]
    mins = (ts.dt.hour*60 + ts.dt.minute).to_numpy()
    dates = ts.dt.date.astype(str).to_numpy()
    n = len(df)

    idx_by_date = {}
    for i, d in enumerate(dates):
        idx_by_date.setdefault(d, []).append(i)

    # eligible signal bars per allowed day (same constraints as replay)
    elig = {}
    for d, idxs in idx_by_date.items():
        if allowed is not None and d not in allowed:
            continue
        ok = [i for i in idxs
              if DEC_START <= mins[i]+5 <= DEC_END and i >= WINDOW_BARS
              and i+1 < n and dates[i+1] == d]
        if ok:
            elig[d] = (ok, idxs[-1])

    def sim_one(i, side, last_idx, d):
        w = slice(i - WINDOW_BARS + 1, i + 1)
        a = _atr_last(h[w], l[w], c[w])
        stop_pts = max(1, int(round(STOP_ATR * a / TICK_SIZE))) * TICK_SIZE
        entry = o[i+1]
        if side == "long":
            stop_px, tgt_px = entry - stop_pts, entry + 2*stop_pts
        else:
            stop_px, tgt_px = entry + stop_pts, entry - 2*stop_pts
        outcome_px = c[last_idx]
        for j in range(i+1, last_idx+1):
            if dates[j] != d: break
            if mins[j] >= FLAT_MIN: outcome_px = o[j]; break
            if side == "long":
                if l[j] <= stop_px: outcome_px = stop_px; break
                if h[j] >= tgt_px:  outcome_px = tgt_px; break
            else:
                if h[j] >= stop_px: outcome_px = stop_px; break
                if l[j] <= tgt_px:  outcome_px = tgt_px; break
        pts = (outcome_px - entry) if side == "long" else (entry - outcome_px)
        return apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS)

    days_sorted = sorted(elig)
    oos_cut = int(len(days_sorted) * IN_SAMPLE_FRACTION)
    totals, oos_totals = [], []
    for _ in range(runs):
        nets = []
        for k, d in enumerate(days_sorted):
            ok, last_idx = elig[d]
            i = ok[rng.integers(len(ok))]
            side = "long" if rng.random() < 0.5 else "short"
            nets.append(sim_one(i, side, last_idx, d))
        totals.append(sum(nets))
        oos_totals.append(sum(nets[oos_cut:]))
    totals = np.array(totals); oos_totals = np.array(oos_totals)
    print(f"  random FULL net$: mean {totals.mean():.0f}  p5 {np.percentile(totals,5):.0f}  "
          f"p50 {np.percentile(totals,50):.0f}  p95 {np.percentile(totals,95):.0f}")
    print(f"  random OOS  net$: mean {oos_totals.mean():.0f}  p5 {np.percentile(oos_totals,5):.0f}  "
          f"p50 {np.percentile(oos_totals,50):.0f}  p95 {np.percentile(oos_totals,95):.0f}")
    print("  -> the strategy only has timing edge if its net is ABOVE p95 of random.")


def run_variants():
    """Bounding/attribution grid: is the negative OOS result robust to the same-bar
    ambiguity assumption, and is the 9:35-10:45 window where the damage happens?"""
    print("="*78)
    print("  VARIANTS — fill-ambiguity bounds + entry-time attribution")
    print("="*78)
    df = load_frame(include_june=True)
    trend_days = trend_no_news_dates(df)
    grid = [
        ("live 9:35+, stop-first",   dict(dec_start=DEC_START, target_first=False)),
        ("live 9:35+, target-first", dict(dec_start=DEC_START, target_first=True)),
        ("entries 10:45+ only",      dict(dec_start=10*60+45, target_first=False)),
        ("entries 10:00+ only",      dict(dec_start=10*60,    target_first=False)),
    ]
    for dayset_name, allowed in [("ALL days", None), ("TREND+no-news", trend_days)]:
        print(f"\n### day filter: {dayset_name}")
        print(HDR)
        for label, kw in grid:
            trades = replay(df, allowed_dates=allowed, **kw)
            nets1 = [t["net1"] for t in trades]
            cut = int(len(nets1) * IN_SAMPLE_FRACTION)
            print_stats_row(stats(nets1, f"full | {label}"))
            print_stats_row(stats(nets1[cut:], f"OOS  | {label}"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    if mode == "parity":
        run_parity()
    elif mode == "full":
        run_full()
    elif mode == "variants":
        run_variants()
    else:
        print("usage: python -m backtest.live_replay [parity|full|variants]")
