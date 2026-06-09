"""
NEWS-DAY EVENT STUDY — can we trade FOMC / NFP days profitably? (user idea 2026-06-09)

Honest framing first:
  - The release CONTENT cannot be known in advance (embargoed). "Get the news before-
    hand" is only possible for the SCHEDULE (we already use it) and consensus forecasts
    (already priced in). Anything else would be insider information.
  - What CAN be tested: mechanical reactions AFTER the release at our speed (5-min
    bars, ~45s polling) — following or fading the initial move.
  - Sample sizes are tiny (~16 FOMC, ~24 NFP days in 2 years). NOTHING here can pass
    the validation gates (100+ trades). This is an exploratory study, not a GO ticket.

Strategies (all: 2.5xATR(14, 8h-window) stop, 2:1 target, flat 15:55, costs applied):
  fomc_follow   : at 14:10 ET (2 bars after the 14:00 statement), trade in the
                  direction of the 13:55->14:10 move.
  fomc_fade     : opposite direction of the same move.
  fomc_break    : first 5-min close outside the 13:00-13:55 pre-range after 14:00,
                  trade the breakout direction.
  nfp_follow    : NFP hits 8:30 (pre-open). At 9:35 trade the direction of the
                  8:25->9:35 reaction.
  nfp_fade      : opposite.
Controls: same entries with RANDOM direction (200 permutations) — isolates whether
the direction rule (not the day/time selection) carries any information.

Run: python -m backtest.news_event
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.live_replay import load_frame, _atr_last, WINDOW_BARS
from backtest.costs import apply_costs
from backtest.news import FOMC_DAYS, nfp_days_for_range
from backtest.config import POINT_VALUE, TICK_SIZE, SLIPPAGE_TICKS

STOP_ATR = 2.5
FLAT_MIN = 15 * 60 + 55


def _day_index(df):
    ts = df["timestamp"]
    mins = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    dates = ts.dt.date.astype(str).to_numpy()
    idx_by_date = {}
    for i, d in enumerate(dates):
        idx_by_date.setdefault(d, []).append(i)
    return mins, dates, idx_by_date


def _bar_at(idx_list, mins, minute):
    """Index of the bar whose window_start == minute, else None."""
    for i in idx_list:
        if mins[i] == minute:
            return i
    return None


def _sim_bracket(df_arrays, i_entry, side, day_idx, mins, dates):
    """Enter at open of bar i_entry, 2.5xATR stop / 2:1 target, flat 15:55."""
    o, h, l, c = df_arrays
    w = slice(i_entry - WINDOW_BARS, i_entry)
    a = _atr_last(h[w], l[w], c[w])
    stop_pts = max(1, int(round(STOP_ATR * a / TICK_SIZE))) * TICK_SIZE
    entry = o[i_entry]
    d = dates[i_entry]
    if side == "long":
        stop_px, tgt_px = entry - stop_pts, entry + 2 * stop_pts
    else:
        stop_px, tgt_px = entry + stop_pts, entry - 2 * stop_pts
    exit_px = c[day_idx[-1]]
    for j in range(i_entry, day_idx[-1] + 1):
        if dates[j] != d:
            break
        if mins[j] >= FLAT_MIN:
            exit_px = o[j]; break
        if side == "long":
            if l[j] <= stop_px: exit_px = stop_px; break
            if h[j] >= tgt_px:  exit_px = tgt_px; break
        else:
            if h[j] >= stop_px: exit_px = stop_px; break
            if l[j] <= tgt_px:  exit_px = tgt_px; break
    pts = (exit_px - entry) if side == "long" else (entry - exit_px)
    return apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS)


def collect_entries(df):
    """Return dict: strategy -> list of (entry_idx, side, day_idx_list)."""
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    mins, dates, idx_by_date = _day_index(df)
    all_dates = sorted(idx_by_date)
    fomc = sorted(set(FOMC_DAYS) & set(all_dates))
    nfp = sorted(nfp_days_for_range(all_dates[0], all_dates[-1]) & set(all_dates))

    out = {k: [] for k in ["fomc_follow", "fomc_fade", "fomc_break",
                           "nfp_follow", "nfp_fade"]}

    for d in fomc:
        idxs = idx_by_date[d]
        i_1355 = _bar_at(idxs, mins, 13 * 60 + 55)   # closes 14:00 (pre-statement)
        i_1410 = _bar_at(idxs, mins, 14 * 60 + 10)   # closes 14:15
        if i_1355 is None or i_1410 is None or i_1410 + 1 >= len(c):
            continue
        move = c[i_1410] - c[i_1355]
        if move != 0:
            side = "long" if move > 0 else "short"
            anti = "short" if move > 0 else "long"
            out["fomc_follow"].append((i_1410 + 1, side, idxs))
            out["fomc_fade"].append((i_1410 + 1, anti, idxs))
        # breakout of the 13:00-13:55 pre-range, first close outside after 14:00
        pre = [i for i in idxs if 13 * 60 <= mins[i] <= 13 * 60 + 55]
        if pre:
            hi, lo = max(h[i] for i in pre), min(l[i] for i in pre)
            for i in idxs:
                if mins[i] < 14 * 60 or i + 1 >= len(c):
                    continue
                if mins[i] >= FLAT_MIN - 30:
                    break
                if c[i] > hi:
                    out["fomc_break"].append((i + 1, "long", idxs)); break
                if c[i] < lo:
                    out["fomc_break"].append((i + 1, "short", idxs)); break

    for d in nfp:
        idxs = idx_by_date[d]
        i_0825 = _bar_at(idxs, mins, 8 * 60 + 25)    # closes 8:30 (pre-release)
        i_0930 = _bar_at(idxs, mins, 9 * 60 + 30)    # closes 9:35
        if i_0825 is None or i_0930 is None or i_0930 + 1 >= len(c):
            continue
        move = c[i_0930] - c[i_0825]
        if move != 0:
            side = "long" if move > 0 else "short"
            anti = "short" if move > 0 else "long"
            out["nfp_follow"].append((i_0930 + 1, side, idxs))
            out["nfp_fade"].append((i_0930 + 1, anti, idxs))
    return out, (o, h, l, c), mins, dates


def main():
    df = load_frame(include_june=True)
    entries, arrays, mins, dates = collect_entries(df)
    rng = np.random.default_rng(11)

    print("=" * 86)
    print("  NEWS-DAY EVENT STUDY — FOMC (14:00 statement) & NFP (8:30 release) reactions")
    print("  EXPLORATORY: n is tiny; nothing here can clear the validation gates.")
    print("=" * 86)
    print(f"  {'strategy':<14}{'n':>4}{'net$':>9}{'exp$':>8}{'PF':>6}{'win%':>6}"
          f"{'rndCtl exp$':>12}{'rnd p95 net$':>13}")

    for name, ents in entries.items():
        if not ents:
            print(f"  {name:<14}   0  (no qualifying days)"); continue
        nets = [_sim_bracket(arrays, i, s, idxs, mins, dates) for (i, s, idxs) in ents]
        a = np.asarray(nets)
        wins = a[a > 0]; gl = -a[a <= 0].sum()
        pf = wins.sum() / gl if gl > 0 else float("inf")
        # random-direction control on the SAME entries
        ctl_tot = []
        for _ in range(200):
            tot = 0.0
            for (i, s, idxs) in ents:
                side = "long" if rng.random() < 0.5 else "short"
                tot += _sim_bracket(arrays, i, side, idxs, mins, dates)
            ctl_tot.append(tot)
        ctl = np.asarray(ctl_tot)
        print(f"  {name:<14}{len(a):>4}{a.sum():>9.0f}{a.mean():>8.2f}{pf:>6.2f}"
              f"{(a > 0).mean()*100:>6.0f}{ctl.mean()/len(a):>12.2f}"
              f"{np.percentile(ctl, 95):>13.0f}")

    print("\n  Read: a strategy is interesting only if net$ > rnd p95 AND the sign of the")
    print("  rule (follow vs fade) is stable across both events. With n=16-24, even a")
    print("  'pass' is weak evidence — it earns a forward paper-trade, not live money.")


if __name__ == "__main__":
    main()
