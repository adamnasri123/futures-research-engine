"""
Modular, parameterized intraday strategy for the broad sweep.

A strategy = (exec_tf, trend_tf, entry method, exit/trail method, stop_atr).
Trend bias comes from an EMA on the (possibly higher) trend timeframe; the
"structure" lives in the entry triggers (breakout/retest of swings) and exits
(swing/chandelier/atr/ema trails). Everything is CAUSAL:

  - indicators at bar i use data up to bar i's close,
  - a signal detected at bar i is FILLED at bar i+1's open,
  - higher-TF bias at exec bar i uses only trend bars that have CLOSED by then,
  - swing points are only used after their K-bar confirmation.

Long and short. One trade per day. Forced flat at SWEEP_FLAT_MIN.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from backtest.strategy import Trade
from backtest.strategy_trend import _atr, _find_swings
from backtest.costs import apply_costs
from backtest.config import (
    POINT_VALUE, SLIPPAGE_TICKS,
    SWEEP_EMA_TREND, SWEEP_EMA_EXEC, SWEEP_ATR_PERIOD, SWEEP_DONCHIAN_N,
    SWEEP_BREAKOUT_N, SWEEP_MOMENTUM_ATR, SWEEP_RETEST_TOL_ATR, SWEEP_TARGET_R,
    SWEEP_ENTRY_END_MIN, SWEEP_FLAT_MIN,
)

ENTRY_METHODS = ["breakout", "donchian", "ema_pullback", "vwap_pullback", "momentum", "retest"]
EXIT_METHODS  = ["swing", "chandelier", "atr_close", "ema", "target_trail"]


@dataclass(frozen=True)
class Params:
    exec_tf:  int     # minutes: 5, 15, 30, 60
    trend_tf: int     # minutes: >= exec_tf
    entry:    str
    exit:     str
    stop_atr: float

    def label(self) -> str:
        return f"{self.exec_tf}m/{self.trend_tf}m {self.entry}->{self.exit} stop{self.stop_atr}"


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def _ema(arr, period):
    out = np.full(len(arr), np.nan)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _vwap(highs, lows, closes, vols):
    typ = (highs + lows + closes) / 3.0
    cpv = np.cumsum(typ * vols)
    cv  = np.cumsum(vols)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cv > 0, cpv / cv, typ)


def _resample_day(day5m: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Aggregate 5-min bars into tf-min bars, anchored at 9:30 ET."""
    if tf == 5:
        return day5m.reset_index(drop=True)
    ts = day5m["timestamp"]
    mins_since_open = (ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)
    bucket = (mins_since_open // tf).to_numpy()
    df = day5m.copy()
    df["_b"] = bucket
    agg = df.groupby("_b").agg(
        timestamp=("timestamp", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index(drop=True)
    return agg


def _bias_series(closes, ema_period):
    """EMA-based trend bias per bar: long / short / None."""
    e = _ema(closes, ema_period)
    bias = [None] * len(closes)
    for i in range(1, len(closes)):
        if np.isnan(e[i]):
            continue
        if closes[i] > e[i] and e[i] > e[i - 1]:
            bias[i] = "long"
        elif closes[i] < e[i] and e[i] < e[i - 1]:
            bias[i] = "short"
    return bias


# ---------------------------------------------------------------------------
# Per-day context (shared by the real run and the matched benchmark)
# ---------------------------------------------------------------------------
class DayCtx:
    __slots__ = ("o", "h", "l", "c", "v", "mins", "atr", "ema", "vwap",
                 "sh", "sl", "allowed", "n", "start_i")

    def __init__(self, day5m, p: Params):
        ex = _resample_day(day5m, p.exec_tf)
        self.o = ex["open"].to_numpy()
        self.h = ex["high"].to_numpy()
        self.l = ex["low"].to_numpy()
        self.c = ex["close"].to_numpy()
        self.v = ex["volume"].to_numpy()
        ts = ex["timestamp"]
        self.mins = ((ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)).to_numpy()  # mins since open
        self.n = len(ex)

        self.atr  = _atr(self.h, self.l, self.c, SWEEP_ATR_PERIOD)
        self.ema  = _ema(self.c, SWEEP_EMA_EXEC)
        self.vwap = _vwap(self.h, self.l, self.c, self.v)
        self.sh, self.sl = _find_swings(self.h, self.l, 2)

        # trend bias mapped onto exec bars (causal)
        if p.trend_tf == p.exec_tf:
            self.allowed = _bias_series(self.c, SWEEP_EMA_TREND)
        else:
            tr = _resample_day(day5m, p.trend_tf)
            tc = tr["close"].to_numpy()
            tts = tr["timestamp"]
            tr_open_min = ((tts.dt.hour * 60 + tts.dt.minute) - (9 * 60 + 30)).to_numpy()
            tr_end_min  = tr_open_min + p.trend_tf
            tbias = _bias_series(tc, SWEEP_EMA_TREND)
            exec_end_min = self.mins + p.exec_tf
            allowed = [None] * self.n
            for i in range(self.n):
                # last trend bar that has CLOSED by this exec bar's close
                j = np.searchsorted(tr_end_min, exec_end_min[i], side="right") - 1
                if j >= 0:
                    allowed[i] = tbias[j]
            self.allowed = allowed

        # earliest bar with valid indicators
        self.start_i = SWEEP_ATR_PERIOD + 1

    def confirmed_swing(self, kind, i):
        """Most recent confirmed swing price of `kind` ('h'/'l') at bar i, or None."""
        src = self.sh if kind == "h" else self.sl
        val = None
        for (idx, px, conf) in src:
            if conf <= i:
                val = px
            else:
                break
        return val


# ---------------------------------------------------------------------------
# Entry triggers (return side 'long'/'short' if the method fires at bar i, else None)
# ---------------------------------------------------------------------------
def _entry_fires(ctx: DayCtx, i: int, method: str):
    side = ctx.allowed[i]
    if side is None:
        return None
    o, h, l, c, atr, ema, vwap = ctx.o, ctx.h, ctx.l, ctx.c, ctx.atr, ctx.ema, ctx.vwap
    if np.isnan(atr[i]):
        return None

    if method == "breakout":
        N = SWEEP_BREAKOUT_N
        if i < N:
            return None
        if side == "long"  and c[i] > h[i-N:i].max():  return "long"
        if side == "short" and c[i] < l[i-N:i].min():  return "short"

    elif method == "donchian":
        N = SWEEP_DONCHIAN_N
        if i < N:
            return None
        if side == "long"  and c[i] > h[i-N:i].max():  return "long"
        if side == "short" and c[i] < l[i-N:i].min():  return "short"

    elif method == "ema_pullback":
        if np.isnan(ema[i]):
            return None
        if side == "long"  and l[i] <= ema[i] and c[i] > ema[i] and c[i] > o[i]:  return "long"
        if side == "short" and h[i] >= ema[i] and c[i] < ema[i] and c[i] < o[i]:  return "short"

    elif method == "vwap_pullback":
        if side == "long"  and l[i] <= vwap[i] and c[i] > vwap[i] and c[i] > o[i]:  return "long"
        if side == "short" and h[i] >= vwap[i] and c[i] < vwap[i] and c[i] < o[i]:  return "short"

    elif method == "momentum":
        rng = h[i] - l[i]
        if rng < SWEEP_MOMENTUM_ATR * atr[i]:
            return None
        if side == "long"  and c[i] > o[i] and c[i] > h[i-1]:  return "long"
        if side == "short" and c[i] < o[i] and c[i] < l[i-1]:  return "short"

    elif method == "retest":
        tol = SWEEP_RETEST_TOL_ATR * atr[i]
        if side == "long":
            lvl = ctx.confirmed_swing("h", i)
            if lvl is not None and c[i] > lvl and l[i] <= lvl + tol:  return "long"
        else:
            lvl = ctx.confirmed_swing("l", i)
            if lvl is not None and c[i] < lvl and h[i] >= lvl - tol:  return "short"

    return None


# ---------------------------------------------------------------------------
# Trade management (returns gross P&L per 1 contract, plus exit reason)
# ---------------------------------------------------------------------------
def _manage(ctx: DayCtx, entry_idx: int, side: str, p: Params):
    o, h, l, c, atr, ema = ctx.o, ctx.h, ctx.l, ctx.c, ctx.atr, ctx.ema
    n = ctx.n
    entry = o[entry_idx]
    a0 = atr[entry_idx - 1] if not np.isnan(atr[entry_idx - 1]) else atr[entry_idx]

    # initial protective stop
    if side == "long":
        stop = entry - p.stop_atr * a0
    else:
        stop = entry + p.stop_atr * a0
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    # target_trail bookkeeping
    half_locked = None
    target = (entry + SWEEP_TARGET_R * risk) if side == "long" else (entry - SWEEP_TARGET_R * risk)

    hh = entry  # highest high since entry (long) / lowest low (short)

    for i in range(entry_idx, n):
        # update running extreme for chandelier
        if side == "long":
            hh = max(hh, h[i])
        else:
            hh = min(hh, l[i])

        # method-specific trailing stop (ratchet only in favorable direction)
        if p.exit == "swing":
            sw = ctx.confirmed_swing("l" if side == "long" else "h", i)
            if sw is not None:
                if side == "long" and sw < c[i]:
                    stop = max(stop, sw)
                elif side == "short" and sw > c[i]:
                    stop = min(stop, sw)
        elif p.exit == "chandelier":
            if side == "long":
                stop = max(stop, hh - p.stop_atr * atr[i])
            else:
                stop = min(stop, hh + p.stop_atr * atr[i])
        elif p.exit == "atr_close":
            if side == "long":
                stop = max(stop, c[i] - p.stop_atr * atr[i])
            else:
                stop = min(stop, c[i] + p.stop_atr * atr[i])
        # "ema" and "target_trail" handled below / via stop too

        # target_trail: lock half at target, then trail remainder via chandelier
        if p.exit == "target_trail" and half_locked is None:
            if side == "long" and h[i] >= target:
                half_locked = target
            elif side == "short" and l[i] <= target:
                half_locked = target
        if p.exit == "target_trail" and half_locked is not None:
            if side == "long":
                stop = max(stop, hh - p.stop_atr * atr[i])
            else:
                stop = min(stop, hh + p.stop_atr * atr[i])

        # stop hit (intrabar)
        if side == "long" and l[i] <= stop:
            return _pnl(side, entry, stop, half_locked), "stop"
        if side == "short" and h[i] >= stop:
            return _pnl(side, entry, stop, half_locked), "stop"

        # EMA exit (on close)
        if p.exit == "ema" and not np.isnan(ema[i]):
            if side == "long" and c[i] < ema[i]:
                return _pnl(side, entry, c[i], half_locked), "ema"
            if side == "short" and c[i] > ema[i]:
                return _pnl(side, entry, c[i], half_locked), "ema"

        # time stop
        if ctx.mins[i] >= SWEEP_FLAT_MIN:
            return _pnl(side, entry, c[i], half_locked), "time_stop"

    return _pnl(side, entry, c[-1], half_locked), "session_end"


def _pnl(side, entry, exit_price, half_locked):
    """P&L per 1 contract in dollars. If half_locked set, half exited at that price."""
    def one(px):
        return (px - entry) if side == "long" else (entry - px)
    if half_locked is None:
        pts = one(exit_price)
    else:
        pts = 0.5 * one(half_locked) + 0.5 * one(exit_price)
    return pts * POINT_VALUE


# ---------------------------------------------------------------------------
# Public: run one day (real strategy) and per-day random outcomes (benchmark)
# ---------------------------------------------------------------------------
def run_ctx(ctx: DayCtx, date: str, p: Params) -> Optional[Trade]:
    """Real strategy on a prebuilt context (ctx depends only on exec_tf/trend_tf)."""
    if ctx.n < ctx.start_i + 2:
        return None
    for i in range(ctx.start_i, ctx.n - 1):
        if ctx.mins[i] > SWEEP_ENTRY_END_MIN:
            break
        side = _entry_fires(ctx, i, p.entry)
        if side is None:
            continue
        res = _manage(ctx, i + 1, side, p)
        if res is None:
            return None
        gross, reason = res
        t = Trade(date=date, side=side, entry_price=ctx.o[i + 1], or_width=0.0)
        t.gross_pnl = gross
        t.exit_reason = reason
        return t
    return None


def random_outcomes_ctx(ctx: DayCtx, p: Params):
    """Net P&L (after costs) for entering at EVERY eligible bar in that bar's
    allowed direction — the direction-matched timing benchmark, per day."""
    if ctx.n < ctx.start_i + 2:
        return []
    out = []
    for i in range(ctx.start_i, ctx.n - 1):
        if ctx.mins[i] > SWEEP_ENTRY_END_MIN:
            break
        side = ctx.allowed[i]
        if side is None:
            continue
        res = _manage(ctx, i + 1, side, p)
        if res is not None:
            out.append(apply_costs(res[0], SLIPPAGE_TICKS))
    return out


def run_day_modular(day5m: pd.DataFrame, date: str, p: Params) -> Optional[Trade]:
    return run_ctx(DayCtx(day5m, p), date, p)
