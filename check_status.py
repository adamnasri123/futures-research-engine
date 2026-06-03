"""
INTRADAY STATUS — run any time during the session to see where you stand
versus the daily risk caps. READ-ONLY (never places or cancels orders).

Usage:
    python check_status.py
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from auth import get_session_token
from accounts import get_accounts
from trades import search_trades
from positions import search_open_positions, POSITION_TYPE
import live_config as cfg


def _et_day_start_utc_iso():
    et = ZoneInfo(cfg.TZ)
    now_et = datetime.now(et)
    start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_et.astimezone(ZoneInfo("UTC"))
    return start_utc.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00")


def main():
    print("=" * 60)
    print(f"  STATUS  —  {datetime.now(ZoneInfo(cfg.TZ)).strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 60)

    token = get_session_token()
    accts = get_accounts(token)
    if not accts:
        print("  No active account found.")
        return
    acct = accts[0]
    aid = acct["id"]
    print(f"  Account : {acct['name']}   Balance ${acct['balance']:,.2f}")

    # --- today's realized P&L ---
    start = _et_day_start_utc_iso()
    trades = search_trades(token, aid, start)
    realized = sum((t.get("profitAndLoss") or 0.0) for t in trades)
    fees = sum((t.get("fees") or 0.0) for t in trades)
    net = realized - fees
    closed = [t for t in trades if t.get("profitAndLoss") is not None]

    print(f"\n  Trades today : {len(trades)} fills ({len(closed)} closed round-turns)")
    print(f"  Realized P&L : ${realized:,.2f}")
    print(f"  Fees         : ${fees:,.2f}")
    print(f"  NET today    : ${net:,.2f}")

    # --- open positions ---
    pos = search_open_positions(token, aid)
    if pos:
        print(f"\n  OPEN positions:")
        for p in pos:
            print(f"    {p['contractId']}  {POSITION_TYPE.get(p['type'], '?')}"
                  f"  size {p['size']}  avg {p['averagePrice']}")
    else:
        print(f"\n  Open positions: none (flat)")

    # --- cap checks ---
    print("\n" + "-" * 60)
    if net <= -cfg.DAILY_LOSS_CAP:
        print(f"  >>> STOP: daily loss cap hit (${net:,.2f} <= -${cfg.DAILY_LOSS_CAP:,.0f}).")
        print("      Close everything and stop trading for the day.")
    elif net >= cfg.DAILY_PROFIT_CAP:
        print(f"  >>> STOP: daily profit cap hit (${net:,.2f} >= +${cfg.DAILY_PROFIT_CAP:,.0f}).")
        print("      Bank it. Stop trading to respect the consistency rule.")
    elif len(closed) >= cfg.MAX_TRADES_PER_DAY:
        print(f"  >>> STOP: max {cfg.MAX_TRADES_PER_DAY} trades reached for the day.")
    else:
        room_loss = cfg.DAILY_LOSS_CAP + net
        room_trades = cfg.MAX_TRADES_PER_DAY - len(closed)
        print(f"  OK to continue. Room to loss cap: ${room_loss:,.2f} | "
              f"trades left: {room_trades}")
    print("-" * 60)


if __name__ == "__main__":
    main()
