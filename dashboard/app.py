"""
Local trading dashboard — READ-ONLY view of the TopStep account + bot, plus a
kill switch.

What it does:
  - Polls the TopStep API (account balance, today's fills, open positions/orders)
    and tails today's bot log. NEVER places or modifies orders.
  - STOP button writes the STOP_BOT flag file; the autotrader sees it on its next
    poll (<= 45s), flattens any open position (live mode) and exits.
  - "Analyst briefing" builds a paste-ready situation report for Claude Code.

Run:  python -m dashboard.app   (then open http://127.0.0.1:8765)
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, send_from_directory

import live_config as cfg
from topstep.auth import get_session_token
from topstep.accounts import get_accounts
from topstep.trades import search_trades
from topstep.positions import search_open_positions
from topstep.orders import search_open_orders, STATUS

ET = ZoneInfo(cfg.TZ)
STATE_FILE = Path(__file__).parent / "state.json"
STOP_FLAG = ROOT / cfg.STOP_FLAG_FILE

app = Flask(__name__, static_folder="static")

_token = {"value": None, "at": 0.0}


def token():
    # session tokens last ~24h; refresh every 50 min to stay safe
    if _token["value"] is None or time.time() - _token["at"] > 3000:
        _token["value"] = get_session_token()
        _token["at"] = time.time()
    return _token["value"]


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"peak_balance": None}


def _save_state(st):
    STATE_FILE.write_text(json.dumps(st, indent=1))


def _day_start_utc_iso():
    now_et = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    return now_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000+00:00")


def _log_tail(n=60):
    path = ROOT / "logs" / f"autotrader_{datetime.now(ET).date().isoformat()}.log"
    if not path.exists():
        return [], None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    mtime = path.stat().st_mtime
    return lines[-n:], mtime


def _bot_running():
    """True if a bot instance is alive: fresh heartbeat, OR (during session hours
    only) today's log shows a session that started and never ended. The log
    fallback covers pre-heartbeat instances but can't see hard kills, so it is
    ignored after 16:05 ET — the bot always exits by ~15:56."""
    hb = ROOT / cfg.HEARTBEAT_FILE
    try:
        if time.time() - hb.stat().st_mtime < cfg.HEARTBEAT_FRESH_SEC:
            return True
    except OSError:
        pass
    now = datetime.now(ET)
    if now.hour * 60 + now.minute >= 16 * 60 + 5:
        return False
    lines, _ = _log_tail(n=500)
    started = ended = -1
    for i, ln in enumerate(lines):
        if "AUTOTRADER START" in ln:
            started = i
        if "AUTOTRADER DONE" in ln or "Standing aside" in ln or "FATAL" in ln \
                or "REFUSED: another autotrader" in ln:
            ended = i
    return started >= 0 and ended < started


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def state():
    try:
        tk = token()
        acct = get_accounts(tk)[0]
        fills = sorted(search_trades(tk, acct["id"], _day_start_utc_iso()),
                       key=lambda t: t.get("creationTimestamp", ""))
        positions = search_open_positions(tk, acct["id"])
        try:
            open_orders = search_open_orders(tk, acct["id"])
        except Exception:
            open_orders = []

        realized = sum((t.get("profitAndLoss") or 0.0) for t in fills)
        fees = sum((t.get("fees") or 0.0) for t in fills)

        st = _load_state()
        bal = acct["balance"]
        if st["peak_balance"] is None or bal > st["peak_balance"]:
            st["peak_balance"] = bal
            _save_state(st)
        mll_floor = st["peak_balance"] - cfg.ACCOUNT_TRAILING_MLL
        cushion = bal - mll_floor

        log_lines, _ = _log_tail()
        running = _bot_running()

        return jsonify({
            "ok": True,
            "now_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET"),
            "account": {"name": acct["name"], "balance": bal,
                        "peak_balance": st["peak_balance"],
                        "mll_floor": mll_floor, "mll_cushion": cushion,
                        "trailing_mll": cfg.ACCOUNT_TRAILING_MLL},
            "today": {"realized": realized, "fees": fees, "net": realized - fees,
                      "loss_cap": cfg.DAILY_LOSS_CAP, "profit_cap": cfg.DAILY_PROFIT_CAP},
            "fills": [{
                "time_et": t.get("creationTimestamp", "")[:19],
                "side": "BUY" if t.get("side") == 0 else "SELL",
                "size": t.get("size"), "price": t.get("price"),
                "pnl": t.get("profitAndLoss"), "fees": t.get("fees"),
            } for t in fills],
            "positions": [{
                "contract": p.get("contractId"), "size": p.get("size"),
                "avg_price": p.get("averagePrice"), "type": p.get("type"),
            } for p in positions],
            "open_orders": [{
                "id": o.get("id"), "type": o.get("type"),
                "status": STATUS.get(o.get("status"), "?"),
                "stop": o.get("stopPrice"), "limit": o.get("limitPrice"),
                "size": o.get("size"),
            } for o in open_orders],
            "bot": {"running": running, "stop_flag": STOP_FLAG.exists(),
                    "live_size": cfg.CONTRACTS_PER_TRADE,
                    "entry_window": f"{cfg.ENTRY_START_MIN//60}:{cfg.ENTRY_START_MIN%60:02d}"
                                    f"-{cfg.ENTRY_END_MIN//60}:{cfg.ENTRY_END_MIN%60:02d} ET"},
            "log": log_lines,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def start():
    """Launch the LIVE bot, detached from the dashboard process. Refuses if an
    instance is already alive (heartbeat or unfinished session in today's log)."""
    if _bot_running():
        return jsonify({"ok": False,
                        "msg": "REFUSED: the bot already appears to be running. "
                               "Stop it first (or wait for it to finish) before starting."})
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()   # a stale stop flag would kill the new instance instantly
    py = ROOT / "venv" / "Scripts" / "python.exe"
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(py), str(ROOT / "autotrader.py"), "--live"],
                     cwd=str(ROOT), creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify({"ok": True,
                    "msg": "Bot launched LIVE (detached — it keeps running if you close "
                           "this page). Watch the log panel; entries only 10:45-12:00 ET."})


@app.route("/api/stop", methods=["POST"])
def stop():
    STOP_FLAG.write_text(f"stopped via dashboard {datetime.now(ET).isoformat()}\n")
    return jsonify({"ok": True, "msg": "STOP flag set. Bot flattens (if live) and exits "
                                       "within ~45s. Remove the flag to allow restarts."})


@app.route("/api/resume", methods=["POST"])
def resume():
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    return jsonify({"ok": True, "msg": "STOP flag removed. The bot will trade again on "
                                       "its next scheduled start (it does not auto-restart)."})


@app.route("/api/briefing")
def briefing():
    s = state().get_json()
    if not s.get("ok"):
        return jsonify({"ok": False, "error": s.get("error")}), 500
    a, t = s["account"], s["today"]
    lines = [
        "ANALYST BRIEFING (paste this to Claude Code in the topstep-bot project)",
        f"As of {s['now_et']}.",
        f"Account {a['name']}: balance ${a['balance']:,.2f}, peak ${a['peak_balance']:,.2f}, "
        f"MLL floor ${a['mll_floor']:,.2f} (cushion ${a['mll_cushion']:,.2f}).",
        f"Today: net ${t['net']:,.2f} (realized ${t['realized']:,.2f}, fees ${t['fees']:,.2f}). "
        f"Caps: -${t['loss_cap']:,.0f} / +${t['profit_cap']:,.0f}.",
        f"Bot running: {s['bot']['running']} | stop flag: {s['bot']['stop_flag']} | "
        f"size {s['bot']['live_size']} micros | entries {s['bot']['entry_window']}.",
        f"Open positions: {s['positions'] or 'none'}",
        f"Today's fills ({len(s['fills'])}):",
    ]
    for f in s["fills"]:
        lines.append(f"  {f['time_et']} {f['side']} {f['size']} @ {f['price']} "
                     f"pnl={f['pnl']} fees={f['fees']}")
    lines.append("Recent log:")
    lines += [f"  {x}" for x in s["log"][-15:]]
    lines.append("Please review per the DAILY ANALYST CHECK-IN PLAYBOOK in logs/ANALYST_JOURNAL.md.")
    return jsonify({"ok": True, "text": "\n".join(lines)})


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8765  (read-only + STOP switch; Ctrl+C to quit)")
    app.run(host="127.0.0.1", port=8765, debug=False)
