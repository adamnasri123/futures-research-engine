"""
MORNING DECISION TOOL — run this each day before/at the open.

It answers ONE question honestly: do we trade today, or stand aside?
Decision rule (the only part with backtest evidence): trade ONLY if today's
daily-regime reads TRENDING and today is NOT a high-impact news day (FOMC/NFP).

It then prints the exact execution rules and risk caps to follow IF trading.
This script is READ-ONLY. It never places an order.

Usage:
    python daily_plan.py
"""
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from topstep.auth import get_session_token
from topstep.accounts import get_accounts
from topstep.contracts import resolve_contract
from topstep.history import get_bars, DAY
from backtest.regime import _adx, _choppiness, ADX_TREND, ADX_CHOP, CHOP_TREND, CHOP_CHOP
from backtest.news import FOMC_DAYS, nfp_days_for_range, FOMC_ANNOUNCE_MIN
import live_config as cfg


def _today_et():
    return datetime.now(ZoneInfo(cfg.TZ)).date()


def _classify_today(token) -> dict:
    """Pull recent daily bars for the front-month and classify today's regime
    using indicators through the last COMPLETED session (causal)."""
    cid = resolve_contract(token, cfg.CONTRACT)
    now = datetime.now(ZoneInfo("UTC"))
    start = (now - timedelta(days=cfg.REGIME_LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    bars = get_bars(token, cid, start, end, unit=DAY, unit_number=1, limit=120, include_partial=False)
    bars = sorted(bars, key=lambda b: b["t"])

    if len(bars) < cfg.REGIME_MIN_BARS:
        return {"ok": False, "n": len(bars)}

    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)

    adx = _adx(h, l, c)
    chop = _choppiness(h, l, c)
    a, ch = adx[-1], chop[-1]

    regime = "neutral"
    if not np.isnan(a) and not np.isnan(ch):
        trend_votes = (a > ADX_TREND) + (ch < CHOP_TREND)
        chop_votes = (a < ADX_CHOP) + (ch > CHOP_CHOP)
        if trend_votes >= 1 and chop_votes == 0:
            regime = "trend"
        elif chop_votes >= 1 and trend_votes == 0:
            regime = "chop"

    return {"ok": True, "n": len(bars), "adx": float(a), "chop": float(ch),
            "regime": regime, "last_session": str(bars[-1]["t"])[:10]}


def _news_today(d) -> set:
    iso = d.isoformat()
    tags = set()
    if iso in FOMC_DAYS:
        tags.add("FOMC (2:00 PM ET)")
    if iso in nfp_days_for_range(iso, iso):
        tags.add("NFP (8:30 AM ET)")
    return tags


def main():
    print("=" * 60)
    print(f"  DAILY TRADING PLAN  —  {_today_et().isoformat()}")
    print("=" * 60)

    token = get_session_token()
    accts = get_accounts(token)
    if accts:
        a = accts[0]
        print(f"  Account : {a['name']}")
        print(f"  Balance : ${a['balance']:,.2f}   canTrade={a['canTrade']}")

    today = _today_et()

    # --- regime ---
    reg = _classify_today(token)
    if not reg["ok"]:
        print(f"\n  Regime  : INSUFFICIENT DATA ({reg['n']} daily bars; need "
              f"{cfg.REGIME_MIN_BARS}). Likely a recent contract roll.")
        print("\n  >>> DECISION: STAND ASIDE (cannot confirm regime safely).")
        return

    print(f"\n  Regime  : {reg['regime'].upper()}  "
          f"(ADX={reg['adx']:.1f}, Choppiness={reg['chop']:.1f}; "
          f"thru {reg['last_session']})")

    # --- news ---
    news = _news_today(today)
    print(f"  News    : {', '.join(news) if news else 'none scheduled'}")

    # --- decision (trend_skipnews) ---
    go = (reg["regime"] == "trend") and (not news)
    print("\n" + "-" * 60)
    if go:
        print("  >>> DECISION: TRADE TODAY (trend day, no high-impact news)")
        _print_rules()
    else:
        why = []
        if reg["regime"] != "trend":
            why.append(f"regime is {reg['regime']}, not trend")
        if news:
            why.append("high-impact news day")
        print(f"  >>> DECISION: STAND ASIDE ({'; '.join(why)})")
        print("\n  No trades today. Protecting capital is the win.")
    print("-" * 60)


def _print_rules():
    risk_pt = cfg.STOP_ATR  # informational
    print(f"""
  EXECUTION RULES (follow mechanically — discipline, not prediction):
    Instrument   : {cfg.CONTRACT}  x {cfg.CONTRACTS_PER_TRADE} contract(s)
    Timeframe    : {cfg.EXEC_TF_MIN}-min bars
    Bias         : LONG if price > {cfg.EMA_TREND}-EMA and EMA rising;
                   SHORT if price < {cfg.EMA_TREND}-EMA and EMA falling
    Entry        : break of last {cfg.BREAKOUT_N}-bar high (long) / low (short),
                   in the bias direction; fill next bar
    Stop         : {cfg.STOP_ATR} x ATR({cfg.ATR_PERIOD}) from entry (protective)
    Exit         : fixed 2:1 target, OR stop hit,
                   OR flat by {cfg.FLAT_MIN//60}:{cfg.FLAT_MIN%60:02d} ET
    Entry window : {cfg.ENTRY_START_MIN//60}:{cfg.ENTRY_START_MIN%60:02d}
                   to {cfg.ENTRY_END_MIN//60}:{cfg.ENTRY_END_MIN%60:02d} ET only

  RISK CAPS (hard — these are the real protection):
    Max trades   : {cfg.MAX_TRADES_PER_DAY} per day
    Daily loss   : STOP at -${cfg.DAILY_LOSS_CAP:,.0f} realized
    Daily profit : STOP at +${cfg.DAILY_PROFIT_CAP:,.0f} realized (consistency rule)
    Hard rule    : 2 losers in a row -> done for the day
    MLL guard    : never let equity fall ${cfg.ACCOUNT_TRAILING_MLL:,.0f} from peak

  Run  python check_status.py  during the session to track P&L vs caps.
""")


if __name__ == "__main__":
    main()
