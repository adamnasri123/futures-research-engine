"""
AUTONOMOUS TRADER — the runtime. Launch once each weekday morning (Task Scheduler).
No human, no LLM in the loop. Everything here is deterministic and mechanical.

SAFETY: DRY-RUN by default. It will NOT place a real order unless you pass --live.
In dry-run it makes every real decision and LOGS the exact order it WOULD send.
Run it dry for several days, read the logs, and only then consider --live.

Honest reminder: the entry signal has NO proven edge (see docs/STRATEGY.md). The risk
caps and guardrails (live_guard.py) are the real protection.

Flow each day:
  1. Decide trade / stand-aside (regime TREND + no FOMC/NFP).  If aside -> log, exit.
  2. Poll completed 5-min bars; compute EMA bias, ATR, 6-bar breakout.
  3. On a valid signal inside the entry window, if preflight passes:
        place ONE bracketed trade (entry+stop+target) — or log it in dry-run.
  4. One-and-done. Keep watching only to force-flat at the cutoff.
  5. At cutoff: ensure flat, write end-of-day summary, exit.

Usage:
  python autotrader.py            # DRY-RUN (default, safe)
  python autotrader.py --live     # places real orders (after dry-run is proven)
"""
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from topstep.auth import get_session_token
from topstep.accounts import get_accounts
from topstep.contracts import resolve_contract
from topstep.history import get_bars, MINUTE
from topstep.orders import BID, ASK
import live_config as cfg
import live_guard as guard
from backtest.regime import _adx, _choppiness, ADX_TREND, ADX_CHOP, CHOP_TREND, CHOP_CHOP
from backtest.news import FOMC_DAYS, nfp_days_for_range
from topstep.history import DAY

POLL_SECONDS = 45        # how often to check for a new completed bar
ET = ZoneInfo(cfg.TZ)

_CID = None


def cid_for(token) -> str:
    """Active front-month contract id, resolved once per run via the API (falls
    back to the static map). Prevents trading/polling an expired contract."""
    global _CID
    if _CID is None:
        _CID = resolve_contract(token, cfg.CONTRACT)
    return _CID

LIVE = "--live" in sys.argv
LOG_PATH = f"logs/autotrader_{datetime.now(ET).date().isoformat()}.log"


# --------------------------------------------------------------------------- #
def log(msg: str):
    stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    line = f"[{stamp}] {msg}"
    print(line)
    try:
        import os
        os.makedirs("logs", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def now_min_et() -> int:
    n = datetime.now(ET)
    return n.hour * 60 + n.minute


# --------------------------------------------------------------------------- #
# Indicators (local, self-contained — live bot does not depend on backtest internals)
# --------------------------------------------------------------------------- #
def ema(arr, period):
    out = np.full(len(arr), np.nan)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def atr(h, l, c, period):
    n = len(h)
    tr = np.empty(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    a = np.full(n, np.nan)
    if n >= period:
        a[period - 1] = tr[:period].mean()
        for i in range(period, n):
            a[i] = (a[i - 1] * (period - 1) + tr[i]) / period
    return a


# --------------------------------------------------------------------------- #
# Day decision (regime + news) — reuses the same logic as daily_plan.py
# --------------------------------------------------------------------------- #
def decide_trade_today(token) -> tuple[bool, str]:
    today_dt = datetime.now(ET)
    if today_dt.weekday() >= 5:
        return False, "weekend (no session)"
    if today_dt.date().isoformat() in cfg.EARLY_CLOSE_DAYS:
        return False, "early-close day (13:00 ET close — force-flat impossible)"

    cid = cid_for(token)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=cfg.REGIME_LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = sorted(get_bars(token, cid, start, end, unit=DAY, unit_number=1, limit=120,
                           include_partial=False), key=lambda b: b["t"])
    if len(bars) < cfg.REGIME_MIN_BARS:
        # Fresh front month after a roll: stitch the previous quarter's daily bars
        # (dates strictly before the new contract's first bar) so ADX/Chop have
        # enough history. One basis gap at the seam — tolerable for regime.
        try:
            from topstep.contracts import previous_quarter
            prev = previous_quarter(cid)
            old = sorted(get_bars(token, prev, start, end, unit=DAY, unit_number=1,
                                  limit=120, include_partial=False), key=lambda b: b["t"])
            first_new = bars[0]["t"] if bars else "9999"
            stitched = [b for b in old if b["t"] < first_new] + bars
            if len(stitched) >= cfg.REGIME_MIN_BARS:
                log(f"Regime history stitched: {len(bars)} bars from {cid} + "
                    f"{len(stitched) - len(bars)} from {prev}")
                bars = stitched
        except Exception as e:
            log(f"Regime stitch failed ({e}) — falling back to short history.")
    if len(bars) < cfg.REGIME_MIN_BARS:
        return False, f"insufficient daily bars ({len(bars)})"

    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    a, ch = _adx(h, l, c)[-1], _choppiness(h, l, c)[-1]

    regime = "neutral"
    if not np.isnan(a) and not np.isnan(ch):
        if ((a > ADX_TREND) + (ch < CHOP_TREND)) >= 1 and ((a < ADX_CHOP) + (ch > CHOP_CHOP)) == 0:
            regime = "trend"
        elif ((a < ADX_CHOP) + (ch > CHOP_CHOP)) >= 1:
            regime = "chop"

    today = datetime.now(ET).date().isoformat()
    news = (today in FOMC_DAYS) or (today in nfp_days_for_range(today, today))

    if regime == "trend" and not news:
        return True, f"TREND (ADX={a:.1f}, Chop={ch:.1f}), no news"
    why = []
    if regime != "trend":
        why.append(f"regime={regime} (ADX={a:.1f}, Chop={ch:.1f})")
    if news:
        why.append("news day")
    return False, "; ".join(why)


# --------------------------------------------------------------------------- #
# Intraday bars + entry signal
# --------------------------------------------------------------------------- #
def recent_5m(token):
    cid = cid_for(token)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = sorted(get_bars(token, cid, start, end, unit=MINUTE, unit_number=cfg.EXEC_TF_MIN,
                           limit=200, include_partial=False), key=lambda b: b["t"])
    return bars


def bars_are_fresh(bars, max_lag_min=15) -> bool:
    """Guard against weekends/holidays/feed freezes: the latest completed bar must
    be recent. An autonomous bot must NEVER act on stale data."""
    if not bars:
        return False
    last_t = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
    lag = (datetime.now(timezone.utc) - last_t).total_seconds() / 60.0
    return lag <= (max_lag_min + cfg.EXEC_TF_MIN)


def signal(bars) -> str | None:
    """Return 'long'/'short'/None using only COMPLETED bars (act on bars[-1])."""
    if len(bars) < max(cfg.ATR_PERIOD, cfg.BREAKOUT_N, cfg.EMA_TREND) + 2:
        return None
    o = np.array([b["o"] for b in bars], float)
    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    e = ema(c, cfg.EMA_TREND)
    i = len(bars) - 1
    if np.isnan(e[i]):
        return None
    up_bias = c[i] > e[i] and e[i] > e[i - 1]
    dn_bias = c[i] < e[i] and e[i] < e[i - 1]
    N = cfg.BREAKOUT_N
    if up_bias and c[i] > h[i - N:i].max():
        return "long"
    if dn_bias and c[i] < l[i - N:i].min():
        return "short"
    return None


def bracket_ticks(bars) -> tuple[int, int]:
    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    a = atr(h, l, c, cfg.ATR_PERIOD)[-1]
    stop_pts = cfg.STOP_ATR * a
    stop_ticks = max(1, int(round(stop_pts / cfg.TICK_SIZE)))
    target_ticks = stop_ticks * 2          # 2R bracket
    return stop_ticks, target_ticks


# --------------------------------------------------------------------------- #
# Single-instance heartbeat — lets the dashboard (and a second launch) know a
# bot is already alive. Touched every poll; stale after HEARTBEAT_FRESH_SEC.
# --------------------------------------------------------------------------- #
def beat():
    try:
        os.makedirs(os.path.dirname(cfg.HEARTBEAT_FILE), exist_ok=True)
        with open(cfg.HEARTBEAT_FILE, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass


def another_instance_running() -> bool:
    try:
        age = time.time() - os.path.getmtime(cfg.HEARTBEAT_FILE)
        return age < cfg.HEARTBEAT_FRESH_SEC
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    if another_instance_running():
        log("REFUSED: another autotrader instance appears to be running "
            f"(heartbeat < {cfg.HEARTBEAT_FRESH_SEC}s old). Exiting.")
        return
    beat()

    mode = "LIVE (placing real orders)" if LIVE else "DRY-RUN (logging only, no orders)"
    log("=" * 60)
    log(f"AUTOTRADER START — mode: {mode}")
    log("=" * 60)

    try:
        token = get_session_token()
        accts = get_accounts(token)
        if not accts:
            log("No active account. Exiting.")
            return
        acct = accts[0]
        aid = acct["id"]
        cid = cid_for(token)
        log(f"Account {acct['name']} | balance ${acct['balance']:,.2f} | {cfg.CONTRACT} ({cid})")

        # --- 1. Day decision ---
        go, why = decide_trade_today(token)
        log(f"Day decision: {'TRADE' if go else 'STAND ASIDE'} — {why}")
        if not go:
            log("Standing aside. No trades today. Exiting.")
            return

        traded = False
        # --- 2. Monitor loop ---
        while True:
            nm = now_min_et()
            beat()

            # Operator kill switch (dashboard STOP button writes this file)
            if os.path.exists(cfg.STOP_FLAG_FILE):
                log("STOP flag detected — flattening (if live) and exiting.")
                if LIVE:
                    st = guard.day_state(token, aid)
                    if st["open_positions"] > 0:
                        from topstep.positions import close_position
                        for p in st["positions"]:
                            close_position(token, aid, p["contractId"])
                            log(f"STOP: closed {p['contractId']} (size {p.get('size')})")
                break

            # Force-flat at cutoff, then we're done
            if nm >= cfg.FLAT_MIN:
                actions = guard.force_flat_if_needed(token, aid) if LIVE else []
                if not LIVE:
                    st = guard.day_state(token, aid)
                    if st["open_positions"] > 0:
                        actions = [f"[dry-run] WOULD force-flat {st['open_positions']} position(s)"]
                for a in actions:
                    log(a)
                log("Past force-flat cutoff. Ending session.")
                break

            # After entry window with no trade — keep running only to guard flat
            if traded or nm > cfg.ENTRY_END_MIN:
                if not traded:
                    log("Entry window closed, no signal today.")
                    break
                time.sleep(POLL_SECONDS)
                continue

            if nm < cfg.ENTRY_START_MIN:
                time.sleep(POLL_SECONDS)
                continue

            # --- inside entry window: look for a signal ---
            try:
                bars = recent_5m(token)
                if not bars_are_fresh(bars):
                    log("Data stale (weekend/holiday/feed freeze) — not trading this tick.")
                    time.sleep(POLL_SECONDS)
                    continue
                sig = signal(bars)
                if sig:
                    ok, reasons = guard.preflight(token, aid)
                    if not ok:
                        log(f"Signal {sig} but preflight BLOCKED: {'; '.join(reasons)}")
                    else:
                        stop_ticks, target_ticks = bracket_ticks(bars)
                        side = BID if sig == "long" else ASK
                        last = bars[-1]["c"]
                        if LIVE:
                            # tag must be UNIQUE per attempt (TopStep rejects reused tags)
                            tag = f"auto-{datetime.now(ET).strftime('%Y%m%d-%H%M%S')}"
                            oid = guard.place_bracketed(token, aid, cid, side,
                                                        stop_ticks, target_ticks,
                                                        custom_tag=tag)
                            log(f"PLACED {sig} {cfg.CONTRACTS_PER_TRADE} {cfg.CONTRACT} @~{last} "
                                f"stop {stop_ticks}t / target {target_ticks}t | orderId {oid}")
                        else:
                            log(f"[dry-run] WOULD place {sig} {cfg.CONTRACTS_PER_TRADE} {cfg.CONTRACT} @~{last} "
                                f"stop {stop_ticks}t / target {target_ticks}t (2R bracket)")
                        traded = True
            except Exception as e:
                log(f"Loop error (continuing): {e}")

            time.sleep(POLL_SECONDS)

        # --- 3. End-of-day summary ---
        st = guard.day_state(token, aid)
        log(f"EOD: net ${st['net']:.2f} | closed trades {st['closed_trades']} | "
            f"open {st['open_positions']}")
        log("AUTOTRADER DONE.")

    except Exception:
        log("FATAL:\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
