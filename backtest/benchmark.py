"""
Random-entry benchmarks — the core honesty check.

A strategy only has a real edge if it clearly beats a random entry that uses the
SAME exit machinery and costs. Two flavours:

  run_benchmark()                — ORB: random direction at a random time.
  direction_matched_benchmark()  — Wave-rider: SAME side the trend rules would pick
                                   (long in uptrends, short in downtrends), but at a
                                   RANDOM entry time. Isolates timing skill from the
                                   direction read and from market beta.

Both share the signature (n_runs, day_groups) and return the same summary dict so
the runner can treat them interchangeably.
"""
import random
import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.costs import apply_costs
from backtest.strategy_trend import _atr, _find_swings
from backtest.config import (
    BENCHMARK_RUNS, TARGET_1_R, TARGET_2_R, OR_MIN_WIDTH_PTS, POINT_VALUE, SLIPPAGE_TICKS,
    TREND_SWING_K, TREND_ATR_PERIOD, TREND_ENTRY_END_MIN, TREND_FLAT_MIN, TREND_MIN_RISK_PTS,
    TREND_IMM_STOP_ATR,
)


def _summary(totals: np.ndarray, **extra) -> dict:
    totals = np.sort(totals)
    out = {
        "mean":         float(totals.mean()),
        "median":       float(totals[len(totals) // 2]),
        "pct_5":        float(totals[int(len(totals) * 0.05)]),
        "pct_95":       float(totals[int(len(totals) * 0.95)]),
        "pct_positive": float((totals > 0).mean()),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# ORB benchmark — random direction, random time, ORB exit machinery
# ---------------------------------------------------------------------------
def _simulate_orb(highs, lows, closes, entry_idx, entry_price, side, or_width) -> float:
    if side == "long":
        stop = entry_price - or_width
        t1   = entry_price + or_width * TARGET_1_R
        t2   = entry_price + or_width * TARGET_2_R
    else:
        stop = entry_price + or_width
        t1   = entry_price - or_width * TARGET_1_R
        t2   = entry_price - or_width * TARGET_2_R

    t1_hit = False
    exit_t1 = exit_t2 = None
    cur_stop = stop
    n = len(highs)

    for i in range(entry_idx, n):
        hi, lo = highs[i], lows[i]
        if side == "long":
            if lo <= cur_stop:
                exit_t1 = exit_t1 if exit_t1 is not None else cur_stop
                exit_t2 = cur_stop
                break
            if not t1_hit and hi >= t1:
                exit_t1 = t1; t1_hit = True; cur_stop = entry_price
            if t1_hit and hi >= t2:
                exit_t2 = t2; break
        else:
            if hi >= cur_stop:
                exit_t1 = exit_t1 if exit_t1 is not None else cur_stop
                exit_t2 = cur_stop
                break
            if not t1_hit and lo <= t1:
                exit_t1 = t1; t1_hit = True; cur_stop = entry_price
            if t1_hit and lo <= t2:
                exit_t2 = t2; break
    else:
        last_close = closes[-1]
        exit_t1 = exit_t1 if exit_t1 is not None else last_close
        exit_t2 = last_close

    if side == "long":
        return ((exit_t1 - entry_price) + (exit_t2 - entry_price)) / 2 * POINT_VALUE
    return ((entry_price - exit_t1) + (entry_price - exit_t2)) / 2 * POINT_VALUE


def run_benchmark(n_runs: int = BENCHMARK_RUNS, day_groups=None) -> dict:
    if day_groups is None:
        day_groups = group_by_day(load_cached())

    per_day_outcomes = []
    for date, day_bars in day_groups:
        if len(day_bars) < 3:
            continue
        highs  = day_bars["high"].to_numpy()
        lows   = day_bars["low"].to_numpy()
        closes = day_bars["close"].to_numpy()
        opens  = day_bars["open"].to_numpy()

        or_width = highs[0] - lows[0]
        if or_width < OR_MIN_WIDTH_PTS:
            continue

        outcomes = []
        for e in range(1, len(day_bars) - 1):
            entry_price = opens[e]
            net_long  = apply_costs(_simulate_orb(highs, lows, closes, e, entry_price, "long",  or_width), SLIPPAGE_TICKS)
            net_short = apply_costs(_simulate_orb(highs, lows, closes, e, entry_price, "short", or_width), SLIPPAGE_TICKS)
            outcomes.append((net_long, net_short))
        if outcomes:
            per_day_outcomes.append(outcomes)

    totals = np.empty(n_runs)
    for r in range(n_runs):
        total = 0.0
        for outcomes in per_day_outcomes:
            net_long, net_short = random.choice(outcomes)
            total += net_long if random.random() < 0.5 else net_short
        totals[r] = total

    return _summary(totals, n_days=len(per_day_outcomes))


# ---------------------------------------------------------------------------
# Wave-rider benchmark — same side as the trend read, random entry time
# ---------------------------------------------------------------------------
def _ride_long(highs, lows, closes, mins, sl, e_idx, entry, resistance):
    init = [px for (idx, px, c) in sl if c <= e_idx and px < entry]
    if not init:
        return None
    trail = max(init)
    if entry - trail < TREND_MIN_RISK_PTS:
        return None
    for i in range(e_idx, len(highs)):
        now = [px for (idx, px, c) in sl if c <= i and px < closes[i]]
        if now:
            ns = max(now)
            if ns > trail:
                trail = ns
        if mins[i] >= TREND_FLAT_MIN:
            return (closes[i] - entry) * POINT_VALUE
        if resistance is not None and highs[i] >= resistance:
            return (resistance - entry) * POINT_VALUE
        if closes[i] < trail:
            return (trail - entry) * POINT_VALUE
    return (closes[-1] - entry) * POINT_VALUE


def _ride_short(highs, lows, closes, mins, sh, e_idx, entry, support):
    init = [px for (idx, px, c) in sh if c <= e_idx and px > entry]
    if not init:
        return None
    trail = min(init)
    if trail - entry < TREND_MIN_RISK_PTS:
        return None
    for i in range(e_idx, len(highs)):
        now = [px for (idx, px, c) in sh if c <= i and px > closes[i]]
        if now:
            ns = min(now)
            if ns < trail:
                trail = ns
        if mins[i] >= TREND_FLAT_MIN:
            return (entry - closes[i]) * POINT_VALUE
        if support is not None and lows[i] <= support:
            return (entry - support) * POINT_VALUE
        if closes[i] > trail:
            return (entry - trail) * POINT_VALUE
    return (entry - closes[-1]) * POINT_VALUE


def direction_matched_benchmark(n_runs: int = BENCHMARK_RUNS, day_groups=None) -> dict:
    if day_groups is None:
        day_groups = group_by_day(load_cached())

    per_day = []
    for date, day_bars in day_groups:
        n = len(day_bars)
        if n < TREND_ATR_PERIOD + 4:
            continue
        opens  = day_bars["open"].to_numpy()
        highs  = day_bars["high"].to_numpy()
        lows   = day_bars["low"].to_numpy()
        closes = day_bars["close"].to_numpy()
        ts     = day_bars["timestamp"]
        mins   = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        atr = _atr(highs, lows, closes, TREND_ATR_PERIOD)
        sh, sl = _find_swings(highs, lows, TREND_SWING_K)

        # the day's trend side at the earliest qualifying bar
        side = None
        for i in range(TREND_ATR_PERIOD, n - 1):
            if mins[i] > TREND_ENTRY_END_MIN:
                break
            if np.isnan(atr[i]):
                continue
            csh = [px for (_, px, c) in sh if c <= i]
            csl = [px for (_, px, c) in sl if c <= i]
            if len(csh) < 2 or len(csl) < 2:
                continue
            if csh[-1] > csh[-2] and csl[-1] > csl[-2]:
                side = "long"; break
            if csh[-1] < csh[-2] and csl[-1] < csl[-2]:
                side = "short"; break
        if side is None:
            continue

        outcomes = []
        for e in range(TREND_ATR_PERIOD, n - 1):
            if mins[e] > TREND_ENTRY_END_MIN:
                break
            entry = opens[e + 1]
            if side == "long":
                overhead = [px for (_, px, c) in sh if c <= e and px > closes[e]]
                res = min(overhead) if overhead else None
                pnl = _ride_long(highs, lows, closes, mins, sl, e + 1, entry, res)
            else:
                below = [px for (_, px, c) in sl if c <= e and px < closes[e]]
                sup = max(below) if below else None
                pnl = _ride_short(highs, lows, closes, mins, sh, e + 1, entry, sup)
            if pnl is not None:
                outcomes.append(apply_costs(pnl, SLIPPAGE_TICKS))
        if outcomes:
            per_day.append(outcomes)

    totals = np.empty(n_runs)
    for r in range(n_runs):
        t = 0.0
        for outcomes in per_day:
            t += random.choice(outcomes)
        totals[r] = t

    return _summary(totals, n_days=len(per_day))


# ---------------------------------------------------------------------------
# "Enter sooner, give it room" benchmark — same side as the trend read,
# random entry time, WIDE ATR stop ridden to the close (matches the strategy).
# ---------------------------------------------------------------------------
def _ride_close_long(highs, lows, closes, mins, e_idx, entry, stop):
    for i in range(e_idx, len(highs)):
        if lows[i] <= stop:
            return (stop - entry) * POINT_VALUE
        if mins[i] >= TREND_FLAT_MIN:
            return (closes[i] - entry) * POINT_VALUE
    return (closes[-1] - entry) * POINT_VALUE


def _ride_close_short(highs, lows, closes, mins, e_idx, entry, stop):
    for i in range(e_idx, len(highs)):
        if highs[i] >= stop:
            return (entry - stop) * POINT_VALUE
        if mins[i] >= TREND_FLAT_MIN:
            return (entry - closes[i]) * POINT_VALUE
    return (entry - closes[-1]) * POINT_VALUE


def direction_matched_benchmark_ride(n_runs: int = BENCHMARK_RUNS, day_groups=None) -> dict:
    if day_groups is None:
        day_groups = group_by_day(load_cached())

    per_day = []
    for date, day_bars in day_groups:
        n = len(day_bars)
        if n < TREND_ATR_PERIOD + 4:
            continue
        opens  = day_bars["open"].to_numpy()
        highs  = day_bars["high"].to_numpy()
        lows   = day_bars["low"].to_numpy()
        closes = day_bars["close"].to_numpy()
        ts     = day_bars["timestamp"]
        mins   = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        atr = _atr(highs, lows, closes, TREND_ATR_PERIOD)
        sh, sl = _find_swings(highs, lows, TREND_SWING_K)

        side = None
        for i in range(TREND_ATR_PERIOD, n - 1):
            if mins[i] > TREND_ENTRY_END_MIN:
                break
            if np.isnan(atr[i]):
                continue
            csh = [px for (_, px, c) in sh if c <= i]
            csl = [px for (_, px, c) in sl if c <= i]
            if len(csh) < 2 or len(csl) < 2:
                continue
            if csh[-1] > csh[-2] and csl[-1] > csl[-2]:
                side = "long"; break
            if csh[-1] < csh[-2] and csl[-1] < csl[-2]:
                side = "short"; break
        if side is None:
            continue

        outcomes = []
        for e in range(TREND_ATR_PERIOD, n - 1):
            if mins[e] > TREND_ENTRY_END_MIN:
                break
            if np.isnan(atr[e]):
                continue
            entry = opens[e + 1]
            if side == "long":
                stop = entry - TREND_IMM_STOP_ATR * atr[e]
                pnl = _ride_close_long(highs, lows, closes, mins, e + 1, entry, stop)
            else:
                stop = entry + TREND_IMM_STOP_ATR * atr[e]
                pnl = _ride_close_short(highs, lows, closes, mins, e + 1, entry, stop)
            outcomes.append(apply_costs(pnl, SLIPPAGE_TICKS))
        if outcomes:
            per_day.append(outcomes)

    totals = np.empty(n_runs)
    for r in range(n_runs):
        t = 0.0
        for outcomes in per_day:
            t += random.choice(outcomes)
        totals[r] = t

    return _summary(totals, n_days=len(per_day))
