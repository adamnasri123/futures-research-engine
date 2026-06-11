"""
1-MINUTE STRUCTURE STOP-AND-REVERSE — user idea 2026-06-10.

Spec (mechanical interpretation of the user's description):
  - 1-min bars. Market structure = fractal swings (k_small bars each side,
    CONFIRMED k_small bars later — no look-ahead). Uptrend = HH+HL; down = LH+LL.
  - Enter with the trend (overnight structure counts, so entries can start 9:30).
  - Stop = most recent confirmed swing low (long) / high (short); TRAILS with
    structure. Signal = CLOSE beyond the trail; fill = NEXT bar open (realistic;
    the old 5-min wave-rider filled AT the level, which flattered it).
  - Take profit = nearest LIQUIDITY: prior-day RTH high/low, overnight high/low,
    or a big-scale swing (k_big fractal = the "major highs/lows" on ~5-min scale).
  - On structure break: optionally REVERSE into the opposite trade (else go flat).
  - After a TP exit: re-enter only once a NEW swing confirms after the exit.
  - Runs all day 9:30-15:55, no new entries after 15:30, force-flat 15:55.
  - Costs on every round turn (commission + 2x slippage ticks).

Also records, for every structure-break reversal entry, how far the breaking
candle CLOSED past the level (in ATR units) — to test the user's hypothesis that
entries far past the level have "missed it".

Run: python -m backtest.structure_sar
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION, CACHE_DIR

CACHE_1MIN = Path(CACHE_DIR) / "ES_1min_24h.parquet"

RTH_START = 9 * 60 + 30
LAST_ENTRY = 15 * 60 + 30
FLAT_MIN = 15 * 60 + 55
ATR_PERIOD = 14


def load_1min():
    df = pd.read_parquet(CACHE_1MIN)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.sort_values("timestamp").reset_index(drop=True)


def _atr(h, l, c, period=ATR_PERIOD):
    n = len(h)
    tr = np.empty(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    a = np.full(n, np.nan)
    if n >= period:
        a[period-1] = tr[:period].mean()
        for i in range(period, n):
            a[i] = (a[i-1]*(period-1) + tr[i]) / period
    return a


def _find_swings(h, l, k):
    """(idx, price, confirm_idx) fractal swings; confirmed k bars later."""
    n = len(h)
    sh, sl = [], []
    for i in range(k, n - k):
        if h[i] == h[i-k:i+k+1].max() and h[i] > h[i-1] and h[i] > h[i+1]:
            sh.append((i, h[i], i + k))
        if l[i] == l[i-k:i+k+1].min() and l[i] < l[i-1] and l[i] < l[i+1]:
            sl.append((i, l[i], i + k))
    return sh, sl


def _swing_arrays(swings):
    """(px, confirm) numpy arrays sorted by confirm index (fractals: already are)."""
    if not swings:
        return np.empty(0), np.empty(0, dtype=int)
    px = np.array([p for (_, p, _) in swings])
    cf = np.array([cf for (_, _, cf) in swings], dtype=int)
    return px, cf


def run_day(ctx, k_small, k_big, reverse, ext_max_atr=None):
    """ctx: dict with arrays for [context_start .. day_end] (overnight + RTH),
    rth0 = index of first RTH bar, pools from prior day. Returns trades list.
    Confirmed-swing queries use pointer advance + numpy (perf)."""
    o, h, l, c, mins = ctx["o"], ctx["h"], ctx["l"], ctx["c"], ctx["mins"]
    n = len(o)
    atr = _atr(h, l, c)
    sh_px, sh_cf = _swing_arrays(_find_swings(h, l, k_small)[0])
    sl_px, sl_cf = _swing_arrays(_find_swings(h, l, k_small)[1])
    bh_px, bh_cf = _swing_arrays(_find_swings(h, l, k_big)[0])
    bl_px, bl_cf = _swing_arrays(_find_swings(h, l, k_big)[1])

    static_above = [p for p in (ctx["pd_high"], ctx["on_high"]) if p is not None]
    static_below = [p for p in (ctx["pd_low"], ctx["on_low"]) if p is not None]

    trades = []
    pos = 0
    entry = stop = 0.0
    tp = None
    last_exit_i = -1
    ext_at_entry = np.nan
    p_sh = p_sl = p_bh = p_bl = 0   # confirmed-prefix pointers

    def pick_tp(side, ref_px, i_bh, i_bl):
        if side > 0:
            big = bh_px[:i_bh]
            cand = [p for p in (list(big[big > ref_px]) + static_above) if p > ref_px]
            return min(cand) if cand else None
        big = bl_px[:i_bl]
        cand = [p for p in (list(big[big < ref_px]) + static_below) if p < ref_px]
        return max(cand) if cand else None

    i = ctx["rth0"]
    while i < n - 1:
        m = mins[i]
        # advance confirmed pointers to bar i
        while p_sh < len(sh_cf) and sh_cf[p_sh] <= i: p_sh += 1
        while p_sl < len(sl_cf) and sl_cf[p_sl] <= i: p_sl += 1
        while p_bh < len(bh_cf) and bh_cf[p_bh] <= i: p_bh += 1
        while p_bl < len(bl_cf) and bl_cf[p_bl] <= i: p_bl += 1

        if m >= FLAT_MIN:
            if pos != 0:
                trades.append({"pts": (o[i] - entry) * pos, "exit": "time",
                               "ext": ext_at_entry})
                pos = 0
            break

        if pos == 0:
            if m <= LAST_ENTRY and p_sh >= 2 and p_sl >= 2 and not np.isnan(atr[i]):
                fresh = last_exit_i < 0 or max(sh_cf[p_sh-1], sl_cf[p_sl-1]) > last_exit_i
                up = sh_px[p_sh-1] > sh_px[p_sh-2] and sl_px[p_sl-1] > sl_px[p_sl-2]
                dn = sh_px[p_sh-1] < sh_px[p_sh-2] and sl_px[p_sl-1] < sl_px[p_sl-2]
                if fresh and (up or dn):
                    side = 1 if up else -1
                    st = sl_px[p_sl-1] if up else sh_px[p_sh-1]
                    if (c[i] - st) * side > 0:
                        pos = side; entry = o[i+1]; stop = st
                        ext_at_entry = np.nan
                        tp = pick_tp(side, entry, p_bh, p_bl)
        else:
            # trail the structure stop
            if pos > 0:
                below = sl_px[:p_sl][sl_px[:p_sl] < c[i]]
                if len(below) and below.max() > stop:
                    stop = below.max()
            else:
                above = sh_px[:p_sh][sh_px[:p_sh] > c[i]]
                if len(above) and above.min() < stop:
                    stop = above.min()

            # take profit at liquidity (limit fill at the level)
            if tp is not None and ((pos > 0 and h[i] >= tp) or (pos < 0 and l[i] <= tp)):
                trades.append({"pts": (tp - entry) * pos, "exit": "liquidity",
                               "ext": ext_at_entry})
                pos = 0; last_exit_i = i
                i += 1; continue

            # structure break on close -> exit (and maybe reverse) at next open
            if (pos > 0 and c[i] < stop) or (pos < 0 and c[i] > stop):
                ext = abs(stop - c[i]) / atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else np.nan
                trades.append({"pts": (o[i+1] - entry) * pos, "exit": "break",
                               "ext": ext_at_entry})
                old = pos
                pos = 0; last_exit_i = i
                if reverse and m <= LAST_ENTRY and (ext_max_atr is None or
                                                    (not np.isnan(ext) and ext <= ext_max_atr)):
                    side = -old
                    st = sl_px[p_sl-1] if side > 0 else sh_px[p_sh-1]
                    if p_sl >= 1 and p_sh >= 1 and (c[i] - st) * side > 0:
                        pos = side; entry = o[i+1]; stop = st
                        ext_at_entry = ext
                        tp = pick_tp(side, entry, p_bh, p_bl)
        i += 1

    else:
        if pos != 0:   # data ended mid-day
            trades.append({"pts": (c[-1] - entry) * pos, "exit": "data_end",
                           "ext": ext_at_entry})
    return trades


def build_contexts(df):
    """Per trading day: arrays from prior-day 18:00 through today 16:00 + pools."""
    ts = df["timestamp"]
    mins = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    dates = ts.dt.date.to_numpy()
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()

    udates = sorted(set(dates))
    date_idx = {d: np.nonzero(dates == d)[0] for d in udates}
    ctxs = []
    for di in range(1, len(udates)):
        d, dprev = udates[di], udates[di - 1]
        idx, idxp = date_idx[d], date_idx[dprev]
        # prior-day RTH pools
        rth_p = idxp[(mins[idxp] >= RTH_START) & (mins[idxp] < 16 * 60)]
        pd_high = float(h[rth_p].max()) if len(rth_p) else None
        pd_low = float(l[rth_p].min()) if len(rth_p) else None
        # overnight = prior day 18:00 -> today 9:30
        on_p = idxp[mins[idxp] >= 18 * 60]
        on_t = idx[mins[idx] < RTH_START]
        on_all = np.concatenate([on_p, on_t])
        on_high = float(h[on_all].max()) if len(on_all) else None
        on_low = float(l[on_all].min()) if len(on_all) else None
        # context window: prior 18:00 .. today end
        win = np.concatenate([on_p, idx])
        rth_mask = mins[win] >= RTH_START
        if not rth_mask.any():
            continue
        rth0 = int(np.argmax((mins[win] >= RTH_START) & (np.isin(win, idx))))
        ctxs.append({"date": str(d), "o": o[win], "h": h[win], "l": l[win],
                     "c": c[win], "mins": mins[win], "rth0": rth0,
                     "pd_high": pd_high, "pd_low": pd_low,
                     "on_high": on_high, "on_low": on_low})
    return ctxs


def stats(nets, label):
    a = np.asarray(nets, float)
    if len(a) == 0:
        return f"  {label:<34} (no trades)"
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = wins.sum() / gl if gl > 0 else float("inf")
    eq = np.cumsum(a); dd = float((eq - np.maximum.accumulate(eq)).min())
    br, cv = simulate_mll(list(a))
    br5, cv5 = simulate_mll(list(a * 5))
    mll = f"1mic:{'Y@%d' % len(cv) if br else 'no'} 5mic:{'Y@%d' % len(cv5) if br5 else 'no'}"
    return (f"  {label:<34}{len(a):>6}{a.sum():>9.0f}{a.mean():>7.2f}{pf:>6.2f}"
            f"{(a > 0).mean()*100:>5.0f}{dd:>9.0f}  {mll}")


def main():
    df = load_1min()
    print(f"1-min bars: {len(df)}  ({df['timestamp'].iloc[0].date()} .. {df['timestamp'].iloc[-1].date()})")
    ctxs = build_contexts(df)
    print(f"trading days: {len(ctxs)}")
    cut_day = int(len(ctxs) * IN_SAMPLE_FRACTION)

    hdr = f"  {'variant':<34}{'n':>6}{'net$':>9}{'exp$':>7}{'PF':>6}{'win%':>5}{'maxDD$':>9}  MLL"
    print("\n" + "=" * 100)
    print("  1-MIN STRUCTURE SAR — full grid (1 micro, costs incl.)   [TRAIN = first 70% of days]")
    print("=" * 100)

    grid = []
    for k_small in (2, 3):
        for k_big in (12, 20):
            for reverse in (True, False):
                grid.append((k_small, k_big, reverse))

    results = {}
    for k_small, k_big, reverse in grid:
        all_nets, all_meta = [], []
        for ci, ctx in enumerate(ctxs):
            trades = run_day(ctx, k_small, k_big, reverse)
            for t in trades:
                all_nets.append(apply_costs(t["pts"] * POINT_VALUE, SLIPPAGE_TICKS))
                all_meta.append({"day": ci, "exit": t["exit"], "ext": t["ext"]})
        results[(k_small, k_big, reverse)] = (all_nets, all_meta)

    print("\n### TRAIN (first 70% of days)")
    print(hdr)
    for key, (nets, meta) in results.items():
        ks, kb, rv = key
        tr_nets = [n for n, m in zip(nets, meta) if m["day"] < cut_day]
        n_days = max(1, cut_day)
        lbl = f"k{ks}/big{kb}/{'SAR' if rv else 'flat-on-break'}"
        print(stats(tr_nets, lbl) + f"  ({len(tr_nets)/n_days:.1f} tr/day)")

    # pick best TRAIN PF with >=200 trades, show OOS once
    def train_pf(key):
        nets, meta = results[key]
        a = np.asarray([n for n, m in zip(nets, meta) if m["day"] < cut_day])
        if len(a) < 200: return -1
        gl = -a[a <= 0].sum()
        return (a[a > 0].sum() / gl) if gl > 0 else -1
    best = max(results, key=train_pf)
    ks, kb, rv = best
    nets, meta = results[best]
    oos = [n for n, m in zip(nets, meta) if m["day"] >= cut_day]
    print(f"\n### BEST-ON-TRAIN -> OOS (touched once): k{ks}/big{kb}/{'SAR' if rv else 'flat'}")
    print(hdr)
    print(stats(oos, "OOS (last 30% of days)"))

    # extension hypothesis: expectancy of REVERSAL entries by how far the breaking
    # candle closed past the level (ATR units) — uses the best variant, TRAIN only
    print("\n### USER HYPOTHESIS: reversal-entry expectancy vs breaking-candle extension (TRAIN)")
    pairs = [(n, m["ext"]) for n, m in zip(nets, meta)
             if m["day"] < cut_day and not (m["ext"] is None or np.isnan(m["ext"]))]
    if len(pairs) >= 40:
        arr = np.array(pairs)
        qs = np.quantile(arr[:, 1], [0.25, 0.5, 0.75])
        bins = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
        names = [f"ext<={qs[0]:.2f}atr", f"..{qs[1]:.2f}", f"..{qs[2]:.2f}", f">{qs[2]:.2f}atr"]
        for (lo, hi), nm in zip(bins, names):
            sel = arr[(arr[:, 1] > lo) & (arr[:, 1] <= hi)][:, 0]
            print(f"  {nm:<16} n={len(sel):>5}  exp ${sel.mean():>7.2f}  win {(sel>0).mean()*100:>3.0f}%")
        print("  (if expectancy decays with extension, the user's 'missed it' intuition is real)")
    else:
        print("  not enough reversal entries to bucket")

    print("\n  HONESTY GATES still apply: a TRAIN winner must survive OOS, beat random,")
    print("  and clear multiple-testing before it means anything. Costs are the enemy")
    print("  of 1-min reversal systems — check exp$ vs the ~$3.12 round-turn cost.")


if __name__ == "__main__":
    main()
