"""
"Catch the wave" — market-structure trend-ride strategy (long-only).

Rules (all mechanical, no discretion):
  - Uptrend  = higher swing highs AND higher swing lows (confirmed structure).
  - Entry    = in an uptrend, price resuming up, with overhead liquidity NOT near
               (nearest resistance > ROOM_ATR_MULT * ATR above, or clear air at highs).
  - Stop     = most recent confirmed swing low (a trailing stop that rises over time).
  - Exit     = close below trailing swing low (structure break)  OR
               price reaches overhead resistance (hits liquidity / take profit)  OR
               15:55 ET time-stop.

No look-ahead: a swing point at bar i is only CONFIRMED at bar i+K, so the entry
loop at bar t may only use swings with confirm_idx <= t.
One trade per day. Long-only (the user described riding uptrends).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from backtest.strategy import Trade   # reuse the same Trade dataclass
from backtest.config import (
    TREND_SWING_K, TREND_ATR_PERIOD, TREND_ROOM_ATR_MULT,
    TREND_ENTRY_END_MIN, TREND_FLAT_MIN, TREND_MIN_RISK_PTS, POINT_VALUE,
    TREND_IMM_STOP_ATR,
)


def _atr(highs, lows, closes, period):
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        # simple rolling mean
        c = np.cumsum(tr)
        atr[period-1] = c[period-1] / period
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period   # Wilder smoothing
    return atr


def _find_swings(highs, lows, k):
    """Return (swing_highs, swing_lows) as lists of (idx, price, confirm_idx)."""
    n = len(highs)
    sh, sl = [], []
    for i in range(k, n - k):
        win_hi = highs[i-k:i+k+1]
        win_lo = lows[i-k:i+k+1]
        if highs[i] == win_hi.max() and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            sh.append((i, highs[i], i + k))
        if lows[i] == win_lo.min() and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            sl.append((i, lows[i], i + k))
    return sh, sl


def run_day_trend(day_bars: pd.DataFrame, date: str) -> Optional[Trade]:
    n = len(day_bars)
    if n < TREND_ATR_PERIOD + 4:
        return None

    opens  = day_bars["open"].to_numpy()
    highs  = day_bars["high"].to_numpy()
    lows   = day_bars["low"].to_numpy()
    closes = day_bars["close"].to_numpy()
    ts     = day_bars["timestamp"]
    mins   = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()

    atr = _atr(highs, lows, closes, TREND_ATR_PERIOD)
    sh, sl = _find_swings(highs, lows, TREND_SWING_K)

    # --- find entry ---
    trade = None
    entry_idx = None
    for i in range(TREND_ATR_PERIOD, n - 1):
        if mins[i] > TREND_ENTRY_END_MIN:
            break
        if np.isnan(atr[i]):
            continue

        # confirmed swings up to bar i
        conf_sh = [(idx, px) for (idx, px, c) in sh if c <= i]
        conf_sl = [(idx, px) for (idx, px, c) in sl if c <= i]
        if len(conf_sh) < 2 or len(conf_sl) < 2:
            continue

        sh1_px = conf_sh[-1][1]; sh0_px = conf_sh[-2][1]   # last two swing highs
        sl1_idx, sl1_px = conf_sl[-1]                       # most recent swing low
        sl0_px = conf_sl[-2][1]

        # uptrend: higher highs AND higher lows
        uptrend = (sh1_px > sh0_px) and (sl1_px > sl0_px)
        if not uptrend:
            continue

        close_i = closes[i]

        # overhead resistance = nearest confirmed swing high ABOVE current close
        overhead = [px for (_, px) in conf_sh if px > close_i]
        resistance = min(overhead) if overhead else None

        # "liquidity not near": clear air above, or resistance far enough to give room
        if resistance is not None and (resistance - close_i) < TREND_ROOM_ATR_MULT * atr[i]:
            continue

        # resumption trigger: price closing up and above prior bar's high (momentum)
        if not (closes[i] > opens[i] and closes[i] > highs[i-1]):
            continue

        # initial stop = most recent confirmed swing low; must be below entry with real risk
        entry_price = opens[i + 1]
        stop_price  = sl1_px
        if entry_price - stop_price < TREND_MIN_RISK_PTS:
            continue

        trade = Trade(
            date=date, side="long",
            entry_price=entry_price, stop_price=stop_price,
            target1_price=(resistance if resistance else 0.0),
            target2_price=(resistance if resistance else 0.0),
            or_width=entry_price - stop_price,   # reuse field = initial risk in pts
        )
        entry_idx = i + 1
        break

    if trade is None:
        return None

    # --- ride the trade ---
    trail_stop = trade.stop_price
    take_profit = trade.target1_price if trade.target1_price > 0 else None

    # precompute confirmed swing lows by confirm index for trailing
    for i in range(entry_idx, n):
        # update trailing stop to most recent confirmed swing low at/under price
        conf_sl_now = [px for (idx, px, c) in sl if c <= i and px < closes[i]]
        if conf_sl_now:
            new_stop = max(conf_sl_now)        # highest swing low below price = tightest valid trail
            if new_stop > trail_stop:
                trail_stop = new_stop

        # time stop
        if mins[i] >= TREND_FLAT_MIN:
            exit_price = closes[i]
            trade.exit_reason = "time_stop"
            break
        # take profit (hit liquidity)
        if take_profit is not None and highs[i] >= take_profit:
            exit_price = take_profit
            trade.exit_reason = "liquidity"
            break
        # structure break (close below trailing swing low)
        if closes[i] < trail_stop:
            exit_price = trail_stop
            trade.exit_reason = "structure_break"
            break
    else:
        exit_price = closes[-1]
        trade.exit_reason = "session_end"

    trade.exit_price_t1 = exit_price
    trade.exit_price_t2 = exit_price
    trade.gross_pnl = (exit_price - trade.entry_price) * POINT_VALUE
    return trade


def run_day_trend_ls(day_bars: pd.DataFrame, date: str) -> Optional[Trade]:
    """
    Long AND short version of the wave-rider. Same structure rules, mirrored:
      - Uptrend   (HH+HL): go LONG,  trail stop UP under swing lows,  TP at resistance.
      - Downtrend (LH+LL): go SHORT, trail stop DOWN over swing highs, TP at support.
    Removes the long-only beta confound: a real structure edge should work both ways.
    """
    n = len(day_bars)
    if n < TREND_ATR_PERIOD + 4:
        return None

    opens  = day_bars["open"].to_numpy()
    highs  = day_bars["high"].to_numpy()
    lows   = day_bars["low"].to_numpy()
    closes = day_bars["close"].to_numpy()
    ts     = day_bars["timestamp"]
    mins   = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()

    atr = _atr(highs, lows, closes, TREND_ATR_PERIOD)
    sh, sl = _find_swings(highs, lows, TREND_SWING_K)

    # --- find entry (first valid long OR short) ---
    trade = None
    entry_idx = None
    for i in range(TREND_ATR_PERIOD, n - 1):
        if mins[i] > TREND_ENTRY_END_MIN:
            break
        if np.isnan(atr[i]):
            continue

        conf_sh = [(idx, px) for (idx, px, c) in sh if c <= i]
        conf_sl = [(idx, px) for (idx, px, c) in sl if c <= i]
        if len(conf_sh) < 2 or len(conf_sl) < 2:
            continue

        sh1_px = conf_sh[-1][1]; sh0_px = conf_sh[-2][1]
        sl1_px = conf_sl[-1][1]; sl0_px = conf_sl[-2][1]
        close_i = closes[i]

        uptrend   = (sh1_px > sh0_px) and (sl1_px > sl0_px)
        downtrend = (sh1_px < sh0_px) and (sl1_px < sl0_px)

        if uptrend:
            # overhead resistance must be far (liquidity not near above)
            overhead = [px for (_, px) in conf_sh if px > close_i]
            resistance = min(overhead) if overhead else None
            if resistance is not None and (resistance - close_i) < TREND_ROOM_ATR_MULT * atr[i]:
                continue
            # resumption up
            if not (closes[i] > opens[i] and closes[i] > highs[i-1]):
                continue
            entry_price = opens[i + 1]
            stop_price  = sl1_px
            if entry_price - stop_price < TREND_MIN_RISK_PTS:
                continue
            trade = Trade(date=date, side="long",
                          entry_price=entry_price, stop_price=stop_price,
                          target1_price=(resistance or 0.0), target2_price=(resistance or 0.0),
                          or_width=entry_price - stop_price)
            entry_idx = i + 1
            break

        elif downtrend:
            # support below must be far (liquidity not near below)
            below = [px for (_, px) in conf_sl if px < close_i]
            support = max(below) if below else None
            if support is not None and (close_i - support) < TREND_ROOM_ATR_MULT * atr[i]:
                continue
            # resumption down
            if not (closes[i] < opens[i] and closes[i] < lows[i-1]):
                continue
            entry_price = opens[i + 1]
            stop_price  = sh1_px   # most recent swing high
            if stop_price - entry_price < TREND_MIN_RISK_PTS:
                continue
            trade = Trade(date=date, side="short",
                          entry_price=entry_price, stop_price=stop_price,
                          target1_price=(support or 0.0), target2_price=(support or 0.0),
                          or_width=stop_price - entry_price)
            entry_idx = i + 1
            break

    if trade is None:
        return None

    # --- ride the trade ---
    take_profit = trade.target1_price if trade.target1_price > 0 else None

    if trade.side == "long":
        trail_stop = trade.stop_price
        for i in range(entry_idx, n):
            conf_now = [px for (idx, px, c) in sl if c <= i and px < closes[i]]
            if conf_now:
                ns = max(conf_now)
                if ns > trail_stop:
                    trail_stop = ns
            if mins[i] >= TREND_FLAT_MIN:
                exit_price = closes[i]; trade.exit_reason = "time_stop"; break
            if take_profit is not None and highs[i] >= take_profit:
                exit_price = take_profit; trade.exit_reason = "liquidity"; break
            if closes[i] < trail_stop:
                exit_price = trail_stop; trade.exit_reason = "structure_break"; break
        else:
            exit_price = closes[-1]; trade.exit_reason = "session_end"
        trade.gross_pnl = (exit_price - trade.entry_price) * POINT_VALUE

    else:  # short
        trail_stop = trade.stop_price
        for i in range(entry_idx, n):
            conf_now = [px for (idx, px, c) in sh if c <= i and px > closes[i]]
            if conf_now:
                ns = min(conf_now)            # lowest swing high above price = tightest valid trail
                if ns < trail_stop:
                    trail_stop = ns
            if mins[i] >= TREND_FLAT_MIN:
                exit_price = closes[i]; trade.exit_reason = "time_stop"; break
            if take_profit is not None and lows[i] <= take_profit:
                exit_price = take_profit; trade.exit_reason = "liquidity"; break
            if closes[i] > trail_stop:
                exit_price = trail_stop; trade.exit_reason = "structure_break"; break
        else:
            exit_price = closes[-1]; trade.exit_reason = "session_end"
        trade.gross_pnl = (trade.entry_price - exit_price) * POINT_VALUE

    trade.exit_price_t1 = exit_price
    trade.exit_price_t2 = exit_price
    return trade


def run_day_trend_immediate(day_bars: pd.DataFrame, date: str) -> Optional[Trade]:
    """
    'Enter sooner, give it room' — tests the finding that the edge is the trend-DAY
    direction read, not the entry trigger.
      - Same trend detection (HH+HL -> long, LH+LL -> short).
      - Enter IMMEDIATELY at the first confirmed-trend bar (no resumption /
        liquidity-not-near filters, which delayed entry and hurt P&L).
      - WIDE ATR protective stop instead of a tight swing trail; NO structure-break
        exit (that's where the chop bled out). Ride to the 15:55 time-stop.
    """
    n = len(day_bars)
    if n < TREND_ATR_PERIOD + 4:
        return None

    opens  = day_bars["open"].to_numpy()
    highs  = day_bars["high"].to_numpy()
    lows   = day_bars["low"].to_numpy()
    closes = day_bars["close"].to_numpy()
    ts     = day_bars["timestamp"]
    mins   = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()

    atr = _atr(highs, lows, closes, TREND_ATR_PERIOD)
    sh, sl = _find_swings(highs, lows, TREND_SWING_K)

    trade = None
    entry_idx = None
    for i in range(TREND_ATR_PERIOD, n - 1):
        if mins[i] > TREND_ENTRY_END_MIN:
            break
        if np.isnan(atr[i]):
            continue
        csh = [px for (_, px, c) in sh if c <= i]
        csl = [px for (_, px, c) in sl if c <= i]
        if len(csh) < 2 or len(csl) < 2:
            continue

        uptrend   = csh[-1] > csh[-2] and csl[-1] > csl[-2]
        downtrend = csh[-1] < csh[-2] and csl[-1] < csl[-2]
        if not (uptrend or downtrend):
            continue

        entry_price = opens[i + 1]
        if uptrend:
            stop_price = entry_price - TREND_IMM_STOP_ATR * atr[i]
            trade = Trade(date=date, side="long", entry_price=entry_price,
                          stop_price=stop_price, or_width=entry_price - stop_price)
        else:
            stop_price = entry_price + TREND_IMM_STOP_ATR * atr[i]
            trade = Trade(date=date, side="short", entry_price=entry_price,
                          stop_price=stop_price, or_width=stop_price - entry_price)
        entry_idx = i + 1
        break

    if trade is None:
        return None

    # --- ride with a fixed wide stop until the close ---
    if trade.side == "long":
        for i in range(entry_idx, n):
            if lows[i] <= trade.stop_price:
                exit_price = trade.stop_price; trade.exit_reason = "stop"; break
            if mins[i] >= TREND_FLAT_MIN:
                exit_price = closes[i]; trade.exit_reason = "time_stop"; break
        else:
            exit_price = closes[-1]; trade.exit_reason = "session_end"
        trade.gross_pnl = (exit_price - trade.entry_price) * POINT_VALUE
    else:
        for i in range(entry_idx, n):
            if highs[i] >= trade.stop_price:
                exit_price = trade.stop_price; trade.exit_reason = "stop"; break
            if mins[i] >= TREND_FLAT_MIN:
                exit_price = closes[i]; trade.exit_reason = "time_stop"; break
        else:
            exit_price = closes[-1]; trade.exit_reason = "session_end"
        trade.gross_pnl = (trade.entry_price - exit_price) * POINT_VALUE

    trade.exit_price_t1 = exit_price
    trade.exit_price_t2 = exit_price
    return trade
