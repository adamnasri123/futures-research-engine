"""
Position-sizing experiment — does varying micros (vs flat 1) help, and at what
risk? Replays the EXACT LIVE strategy (10-EMA bias + 6-bar breakout, fixed 2.5xATR
stop, 2:1 target, 9:35-12:00 entry, 15:55 flat) over the 507 cached days, producing
ONE chronological trade list. Then applies each sizing scheme to that same list.

HONEST PRINCIPLE: sizing does NOT change which trades happen or their per-contract
P&L. It only multiplies contract count. So it cannot create edge — it only reshapes
the risk/return distribution. We measure: total net, max drawdown, and (the real
question) does it breach the $2,000 trailing MLL.

Schemes:
  flat        : always 1 micro (baseline = current live behavior)
  atr_vol     : size to hold ~constant $ risk (fewer micros when stop is wide)
  regime_adx  : bigger when daily ADX is strong (2 if >30, 3 if >40)
  anti_mart   : start 1; +1 after a win (cap), reset to 1 after a loss

All evaluated OUT-OF-SAMPLE (last 30% of days) + walk-forward-style full curve, with
the same costs and the $2000 MLL check.

Run: python -m backtest.sizing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.regime import classify
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import (
    POINT_VALUE, TICK_SIZE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION, TRAILING_MLL,
)

# Mirror the LIVE strategy params (live_config.py)
EMA_TREND   = 10
BREAKOUT_N  = 6
ATR_PERIOD  = 14
STOP_ATR    = 2.5
TARGET_MULT = 2.0           # target distance = 2x stop distance
ENTRY_START = 9 * 60 + 35
ENTRY_END   = 12 * 60
FLAT_MIN    = 15 * 60 + 55

MAX_CONTRACTS  = 4          # hard cap for any sizing scheme
ATR_TARGET_USD = 50.0       # atr_vol scheme aims for ~this $ risk per trade


def _ema(arr, period):
    out = np.full(len(arr), np.nan)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


def _atr(h, l, c, period):
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


def replay_live_trades(day_groups):
    """Return chronological list of trades. Each trade is a dict with:
       date, side, pts (per-contract P&L in points, gross), stop_pts (risk distance),
       outcome ('target'/'stop'/'time'). Mirrors the live signal + fixed bracket."""
    trades = []
    for date, day in day_groups:
        o = day["open"].to_numpy(); h = day["high"].to_numpy()
        l = day["low"].to_numpy();  c = day["close"].to_numpy()
        ts = day["timestamp"]
        mins = (ts.dt.hour*60 + ts.dt.minute).to_numpy()
        n = len(day)
        if n < max(ATR_PERIOD, BREAKOUT_N, EMA_TREND) + 3:
            continue
        ema = _ema(c, EMA_TREND)
        atr = _atr(h, l, c, ATR_PERIOD)

        entered = False
        for i in range(max(ATR_PERIOD, BREAKOUT_N, EMA_TREND)+1, n-1):
            if mins[i] < ENTRY_START or mins[i] > ENTRY_END:
                continue
            if np.isnan(ema[i]) or np.isnan(atr[i]):
                continue
            up = c[i] > ema[i] and ema[i] > ema[i-1]
            dn = c[i] < ema[i] and ema[i] < ema[i-1]
            sig = None
            if up and c[i] > h[i-BREAKOUT_N:i].max():
                sig = "long"
            elif dn and c[i] < l[i-BREAKOUT_N:i].min():
                sig = "short"
            if sig is None:
                continue

            # enter next bar open
            entry = o[i+1]
            a = atr[i]
            stop_pts = STOP_ATR * a
            tgt_pts  = TARGET_MULT * stop_pts
            if sig == "long":
                stop_px = entry - stop_pts; tgt_px = entry + tgt_pts
            else:
                stop_px = entry + stop_pts; tgt_px = entry - tgt_pts

            outcome, exit_px = "time", c[-1]
            for j in range(i+1, n):
                if mins[j] >= FLAT_MIN:
                    outcome, exit_px = "time", c[j]; break
                if sig == "long":
                    if l[j] <= stop_px: outcome, exit_px = "stop", stop_px; break
                    if h[j] >= tgt_px:  outcome, exit_px = "target", tgt_px; break
                else:
                    if h[j] >= stop_px: outcome, exit_px = "stop", stop_px; break
                    if l[j] <= tgt_px:  outcome, exit_px = "target", tgt_px; break

            pts = (exit_px - entry) if sig == "long" else (entry - exit_px)
            trades.append({"date": date, "side": sig, "pts": pts,
                           "stop_pts": stop_pts, "outcome": outcome})
            entered = True
            break  # one trade/day
    return trades


def _net_per_contract(pts):
    """Gross points -> net $ for ONE contract (incl. commission + slippage)."""
    return apply_costs(pts * POINT_VALUE, SLIPPAGE_TICKS)


def size_flat(trade, prev_size, prev_pnl, adx):
    return 1


def size_atr_vol(trade, prev_size, prev_pnl, adx):
    risk_per_contract = trade["stop_pts"] * POINT_VALUE
    if risk_per_contract <= 0:
        return 1
    n = round(ATR_TARGET_USD / risk_per_contract)
    return int(max(1, min(MAX_CONTRACTS, n)))


def size_regime_adx(trade, prev_size, prev_pnl, adx):
    if adx is None or np.isnan(adx):
        return 1
    if adx > 40: return min(MAX_CONTRACTS, 3)
    if adx > 30: return 2
    return 1


def size_anti_mart(trade, prev_size, prev_pnl, adx):
    if prev_pnl is None:
        return 1
    if prev_pnl > 0:
        return int(min(MAX_CONTRACTS, prev_size + 1))
    return 1


SCHEMES = {
    "flat":       size_flat,
    "atr_vol":    size_atr_vol,
    "regime_adx": size_regime_adx,
    "anti_mart":  size_anti_mart,
}


def apply_scheme(trades, scheme_fn, regime_map):
    """Return list of net $ P&L per trade after sizing."""
    nets = []
    prev_size, prev_pnl = 1, None
    for t in trades:
        adx = regime_map.get(t["date"], {}).get("adx")
        size = scheme_fn(t, prev_size, prev_pnl, adx)
        per1 = _net_per_contract(t["pts"])
        pnl = per1 * size
        nets.append(pnl)
        prev_size, prev_pnl = size, pnl
    return nets


def stats(nets):
    n = len(nets)
    if n == 0:
        return None
    a = np.asarray(nets, float)
    wins = a[a > 0]; gl = -a[a <= 0].sum()
    pf = (wins.sum()/gl) if gl > 0 else float("inf")
    eq = np.cumsum(a); peak = np.maximum.accumulate(eq)
    maxdd = float((eq - peak).min())
    breach, _ = simulate_mll(list(a))
    return {"n": n, "net": float(a.sum()), "exp": float(a.mean()),
            "pf": pf, "wr": len(wins)/n, "maxdd": maxdd, "breach": breach,
            "avg_size": None}


def main():
    print("="*70)
    print("  POSITION SIZING TEST — live strategy, 4 schemes vs flat-1-micro")
    print("="*70)

    groups = group_by_day(load_cached())
    regime_map = classify(groups)
    trades = replay_live_trades(groups)
    print(f"\nReplayed live strategy: {len(trades)} trades over {len(groups)} days")
    if not trades:
        print("No trades — abort."); return

    # chronological OOS split
    cut = int(len(trades) * IN_SAMPLE_FRACTION)
    oos = trades[cut:]
    print(f"Out-of-sample slice: {len(oos)} trades (last {100-int(IN_SAMPLE_FRACTION*100)}%)")

    # report avg contract size per scheme (informational)
    def avg_size(trades_sub, fn):
        sizes = []; ps, pp = 1, None
        for t in trades_sub:
            adx = regime_map.get(t["date"], {}).get("adx")
            s = fn(t, ps, pp, adx)
            sizes.append(s)
            per1 = _net_per_contract(t["pts"]); pp = per1 * s; ps = s
        return np.mean(sizes)

    print("\n--- FULL PERIOD ---")
    _table(trades, regime_map, avg_size)
    print("\n--- OUT-OF-SAMPLE (the honest read) ---")
    _table(oos, regime_map, avg_size)

    print("\n" + "="*70)
    print("  READING THIS:")
    print("  - 'flat' is the baseline (current live = 1 micro).")
    print("  - If baseline expectancy is ~0/negative, EVERY scheme just scales that.")
    print("  - The real question is MLL breach + drawdown, not the headline net.")
    print("  - A scheme 'wins' only if it improves risk WITHOUT breaching MLL.")
    print("="*70)


def _table(trades_sub, regime_map, avg_size_fn):
    hdr = f"  {'scheme':<12} {'n':>4} {'avgSz':>6} {'net$':>9} {'exp$':>7} {'PF':>5} {'win%':>5} {'maxDD$':>9} {'MLLbreach':>10}"
    print(hdr)
    print("  " + "-"*len(hdr))
    for name, fn in SCHEMES.items():
        nets = apply_scheme(trades_sub, fn, regime_map)
        s = stats(nets)
        asz = avg_size_fn(trades_sub, fn)
        print(f"  {name:<12} {s['n']:>4} {asz:>6.2f} {s['net']:>9.0f} {s['exp']:>7.2f} "
              f"{s['pf']:>5.2f} {s['wr']*100:>5.0f} {s['maxdd']:>9.0f} "
              f"{('YES' if s['breach'] else 'no'):>10}")


if __name__ == "__main__":
    main()
