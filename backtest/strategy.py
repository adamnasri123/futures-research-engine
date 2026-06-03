"""
Phase B: 5-Minute ORB strategy logic.
Pure function — bars in, trade out. No network, no global state.

Opening Range = 9:30 bar (first 5-min candle of RTH session).
Entry signal on close of that bar vs OR levels — fill on NEXT bar open (no look-ahead).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from backtest.config import OR_MIN_WIDTH_PTS, MAX_VWAP_CROSSES, TARGET_1_R, TARGET_2_R, POINT_VALUE


@dataclass
class Trade:
    date:            str
    side:            str
    entry_price:     float = 0.0
    stop_price:      float = 0.0
    target1_price:   float = 0.0
    target2_price:   float = 0.0
    or_width:        float = 0.0
    exit_price_t1:   float = 0.0
    exit_price_t2:   float = 0.0
    exit_reason:     str   = ""
    gross_pnl:       float = 0.0
    net_pnl:         float = 0.0
    skipped:         bool  = False
    skip_reason:     str   = ""


def _vwap_series(highs, lows, closes, volumes):
    """
    Session-anchored running VWAP as a numpy array.
    vwap[i] = cumulative VWAP using bars 0..i (inclusive).
    Computed in O(n) with cumulative sums (was O(n^2) before).
    """
    typical = (highs + lows + closes) / 3.0
    cum_pv  = np.cumsum(typical * volumes)
    cum_v   = np.cumsum(volumes)
    # avoid divide-by-zero: where cum_v == 0, fall back to typical price
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cum_v > 0, cum_pv / cum_v, typical)
    return vwap


def _count_vwap_crosses(closes, vwap, up_to_idx: int) -> int:
    """Count how many times close crosses VWAP in the first `up_to_idx` bars."""
    crosses = 0
    prev_side = None
    for i in range(up_to_idx):
        side = "above" if closes[i] > vwap[i] else "below"
        if prev_side is not None and side != prev_side:
            crosses += 1
        prev_side = side
    return crosses


def run_day(day_bars: pd.DataFrame, date: str) -> Optional[Trade]:
    """
    Given all RTH 5-min bars for one day, return the trade or None.
    Bars must be sorted by timestamp ascending.
    Expected layout: bar[0]=9:30, bar[1]=9:35, ..., bar[23]=11:55 ET.
    """
    n = len(day_bars)
    if n < 3:
        return None

    # Convert to numpy arrays once — fast indexing in the simulation loops.
    opens   = day_bars["open"].to_numpy()
    highs   = day_bars["high"].to_numpy()
    lows    = day_bars["low"].to_numpy()
    closes  = day_bars["close"].to_numpy()
    volumes = day_bars["volume"].to_numpy()

    # --- Opening Range: first bar (9:30:00) ---
    or_high  = highs[0]
    or_low   = lows[0]
    or_width = or_high - or_low

    if or_width < OR_MIN_WIDTH_PTS:
        return Trade(date=date, side="", or_width=or_width, skipped=True,
                     skip_reason=f"OR width {or_width:.2f} < min {OR_MIN_WIDTH_PTS}")

    vwap = _vwap_series(highs, lows, closes, volumes)

    # VWAP crosses in first 12 bars (first hour = 9:30–10:25)
    first_hour_bars = min(12, n)
    vwap_crosses = _count_vwap_crosses(closes, vwap, first_hour_bars)
    if vwap_crosses > MAX_VWAP_CROSSES:
        return Trade(date=date, side="", or_width=or_width, skipped=True,
                     skip_reason=f"Choppy: {vwap_crosses} VWAP crosses in first hour")

    # --- Signal: breakout on close of any bar AFTER the OR bar ---
    # Fill on the OPEN of the FOLLOWING bar (no look-ahead bias)
    trade = None
    entry_bar_idx = None
    for i in range(1, n - 1):
        if closes[i] > or_high and closes[i] > vwap[i]:
            side = "long"
        elif closes[i] < or_low and closes[i] < vwap[i]:
            side = "short"
        else:
            continue

        entry_price = opens[i + 1]   # next bar open

        if side == "long":
            stop_price    = or_low
            target1_price = entry_price + or_width * TARGET_1_R
            target2_price = entry_price + or_width * TARGET_2_R
        else:
            stop_price    = or_high
            target1_price = entry_price - or_width * TARGET_1_R
            target2_price = entry_price - or_width * TARGET_2_R

        trade = Trade(
            date=date, side=side,
            entry_price=entry_price, stop_price=stop_price,
            target1_price=target1_price, target2_price=target2_price,
            or_width=or_width,
        )
        entry_bar_idx = i + 1
        break   # max 1 trade per day

    if trade is None:
        return None

    # --- Simulate trade through remaining bars ---
    t1_hit = False
    current_stop = trade.stop_price

    for i in range(entry_bar_idx, n):
        if trade.side == "long":
            if lows[i] <= current_stop:
                trade.exit_price_t1 = trade.exit_price_t1 or current_stop
                trade.exit_price_t2 = current_stop
                trade.exit_reason = "stop" if not t1_hit else "stop_t2"
                break
            if not t1_hit and highs[i] >= trade.target1_price:
                trade.exit_price_t1 = trade.target1_price
                t1_hit = True
                current_stop = trade.entry_price   # move stop to breakeven
            if t1_hit and highs[i] >= trade.target2_price:
                trade.exit_price_t2 = trade.target2_price
                trade.exit_reason = "target"
                break
        else:
            if highs[i] >= current_stop:
                trade.exit_price_t1 = trade.exit_price_t1 or current_stop
                trade.exit_price_t2 = current_stop
                trade.exit_reason = "stop" if not t1_hit else "stop_t2"
                break
            if not t1_hit and lows[i] <= trade.target1_price:
                trade.exit_price_t1 = trade.target1_price
                t1_hit = True
                current_stop = trade.entry_price
            if t1_hit and lows[i] <= trade.target2_price:
                trade.exit_price_t2 = trade.target2_price
                trade.exit_reason = "target"
                break
    else:
        # Time stop
        last_close = closes[-1]
        trade.exit_price_t1 = trade.exit_price_t1 or last_close
        trade.exit_price_t2 = last_close
        trade.exit_reason = "time_stop"

    # --- Gross P&L (half position at T1, half at T2) ---
    if trade.side == "long":
        pnl_t1 = (trade.exit_price_t1 - trade.entry_price) * POINT_VALUE
        pnl_t2 = (trade.exit_price_t2 - trade.entry_price) * POINT_VALUE
    else:
        pnl_t1 = (trade.entry_price - trade.exit_price_t1) * POINT_VALUE
        pnl_t2 = (trade.entry_price - trade.exit_price_t2) * POINT_VALUE

    trade.gross_pnl = (pnl_t1 + pnl_t2) / 2

    return trade
