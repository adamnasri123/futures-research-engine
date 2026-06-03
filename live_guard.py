"""
GUARDRAIL LAYER — the rules enforced IN CODE on every trade.

Nothing here predicts the market. This module's only job is to make it
*impossible* to break the risk rules: no naked orders, no trading past the daily
loss limit, no second trade after the one-trade-per-day flag, force-flat at the
cutoff. The strategy may have no edge; these guardrails are what keep a no-edge
system from doing real damage during the eval.

Every function is defensive. When in doubt, it BLOCKS.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import time

from accounts import get_accounts
from trades import search_trades
from positions import search_open_positions, close_position
from orders import place_order, search_open_orders, STOP, TRAILING_STOP, MARKET, BID, ASK
import live_config as cfg


# ---------------------------------------------------------------------------
# Day P&L / state
# ---------------------------------------------------------------------------
def _et_day_start_iso() -> str:
    et = ZoneInfo(cfg.TZ)
    now_et = datetime.now(et).replace(hour=0, minute=0, second=0, microsecond=0)
    return now_et.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.000000+00:00")


def day_state(token: str, account_id: int) -> dict:
    """Today's realized net P&L, closed-trade count, and open-position count."""
    trades = search_trades(token, account_id, _et_day_start_iso())
    realized = sum((t.get("profitAndLoss") or 0.0) for t in trades)
    fees = sum((t.get("fees") or 0.0) for t in trades)
    closed = [t for t in trades if t.get("profitAndLoss") is not None]
    pos = search_open_positions(token, account_id)
    return {
        "net": realized - fees,
        "closed_trades": len(closed),
        "open_positions": len(pos),
        "positions": pos,
    }


def _now_min_et() -> int:
    now = datetime.now(ZoneInfo(cfg.TZ))
    return now.hour * 60 + now.minute


# ---------------------------------------------------------------------------
# Pre-trade preflight — returns (allowed: bool, reasons: list[str])
# ---------------------------------------------------------------------------
def preflight(token: str, account_id: int) -> tuple[bool, list[str]]:
    reasons = []
    st = day_state(token, account_id)
    now_min = _now_min_et()

    # 1. Daily loss limit guard — block well BEFORE the cap (one trade's risk of buffer)
    per_trade_risk = cfg.STOP_ATR * 6 * cfg.POINT_VALUE * cfg.CONTRACTS_PER_TRADE  # rough max
    if st["net"] <= -(cfg.DAILY_LOSS_CAP - per_trade_risk):
        reasons.append(f"near/at daily loss cap (net ${st['net']:.0f}, cap -${cfg.DAILY_LOSS_CAP:.0f})")

    # 2. Daily profit cap (consistency rule)
    if st["net"] >= cfg.DAILY_PROFIT_CAP:
        reasons.append(f"daily profit cap hit (net ${st['net']:.0f})")

    # 3. One-trade-per-day / max trades
    if st["closed_trades"] >= cfg.MAX_TRADES_PER_DAY:
        reasons.append(f"max trades reached ({st['closed_trades']}/{cfg.MAX_TRADES_PER_DAY})")

    # 4. Already in a position
    if st["open_positions"] > 0:
        reasons.append("a position is already open")

    # 5. Entry window
    if now_min < cfg.ENTRY_START_MIN:
        reasons.append(f"before entry window opens ({cfg.ENTRY_START_MIN//60}:{cfg.ENTRY_START_MIN%60:02d} ET)")
    if now_min > cfg.ENTRY_END_MIN:
        reasons.append(f"past entry cutoff ({cfg.ENTRY_END_MIN//60}:{cfg.ENTRY_END_MIN%60:02d} ET)")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Bracketed entry — REFUSES to send a naked order
# ---------------------------------------------------------------------------
def place_bracketed(token: str, account_id: int, contract_id: str, side: int,
                    stop_ticks: int, target_ticks: int,
                    size: int = None, custom_tag: str = None) -> int:
    """Place a MARKET entry WITH an attached stop-loss and take-profit bracket.
    Raises if either bracket leg is missing — there is no code path to a naked order.
    Runs preflight first and refuses if any risk rule is violated."""
    if size is None:
        size = cfg.CONTRACTS_PER_TRADE

    if stop_ticks is None or stop_ticks <= 0:
        raise ValueError("REFUSED: stop_ticks must be > 0 — no naked orders.")
    if target_ticks is None or target_ticks <= 0:
        raise ValueError("REFUSED: target_ticks must be > 0 — no naked orders.")
    if size < 1 or size > cfg.MAX_CONTRACTS_HARD:
        raise ValueError(f"REFUSED: size must be 1..{cfg.MAX_CONTRACTS_HARD} (got {size}).")

    allowed, reasons = preflight(token, account_id)
    if not allowed:
        raise RuntimeError("REFUSED by preflight: " + "; ".join(reasons))

    order_id = place_order(
        token, account_id, contract_id,
        type=MARKET, side=side, size=size,
        stop_loss_ticks=stop_ticks, take_profit_ticks=target_ticks,
        custom_tag=custom_tag,
    )

    # POST-ENTRY VERIFICATION: confirm a protective stop actually attached on the
    # server. The bracket order format has never been live-tested; if the stop leg
    # silently failed we'd be holding a naked position. If we can't confirm a stop
    # within a few seconds, flatten immediately — "never naked" must hold even if
    # the bracket API behaves differently than the docs.
    if not _has_protective_stop(token, account_id, contract_id):
        time.sleep(2)  # allow the bracket legs a moment to register
        if not _has_protective_stop(token, account_id, contract_id):
            try:
                close_position(token, account_id, contract_id)
            finally:
                raise RuntimeError(
                    "ABORTED: no protective stop detected after entry; position "
                    "force-closed. Do NOT go live again until the bracket format "
                    "is confirmed against TopStep."
                )

    return order_id


def _has_protective_stop(token: str, account_id: int, contract_id: str) -> bool:
    """True if an open STOP (or trailing-stop) order exists for this contract."""
    try:
        open_orders = search_open_orders(token, account_id)
    except Exception:
        return False
    for o in open_orders:
        if o.get("contractId") == contract_id and o.get("type") in (STOP, TRAILING_STOP):
            return True
    return False


# ---------------------------------------------------------------------------
# Force-flat at the cutoff
# ---------------------------------------------------------------------------
def force_flat_if_needed(token: str, account_id: int) -> list[str]:
    """If we're at/past the flat cutoff and a position is open, close it at market.
    Returns a list of human-readable actions taken."""
    actions = []
    if _now_min_et() < cfg.FLAT_MIN:
        return actions
    from positions import close_position
    pos = search_open_positions(token, account_id)
    for p in pos:
        close_position(token, account_id, p["contractId"])
        actions.append(f"force-flat: closed {p['contractId']} (size {p['size']})")
    return actions


if __name__ == "__main__":
    # Dry status: show preflight verdict without placing anything.
    from auth import get_session_token
    token = get_session_token()
    accts = get_accounts(token)
    if not accts:
        print("No account.")
    else:
        aid = accts[0]["id"]
        st = day_state(token, aid)
        ok, reasons = preflight(token, aid)
        print(f"Account {accts[0]['name']}")
        print(f"  net today ${st['net']:.2f} | closed {st['closed_trades']} | open {st['open_positions']}")
        print(f"  preflight: {'CLEAR to place a bracketed trade' if ok else 'BLOCKED'}")
        for r in reasons:
            print(f"    - {r}")
