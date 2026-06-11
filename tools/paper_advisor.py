"""
PAPER ADVISOR — the "script asks Claude for a read" experiment. PAPER ONLY.

This tool NEVER places orders. It exists to MEASURE whether an LLM analyst read
adds value over the mechanical baseline:

  1. `python -m tools.paper_advisor ask`
     Builds a live market snapshot (price, gap, prior-day/overnight levels, regime,
     vol context), composes a decision prompt, and:
       - if the `claude` CLI is installed: runs it headlessly and logs the decision;
       - otherwise: prints the prompt to paste into Claude Code, then logs the
         pasted JSON via `python -m tools.paper_advisor log '<json>'`.
     Decisions are appended to logs/advisor_decisions.jsonl with outcome=null.

  2. `python -m tools.paper_advisor score`
     For each logged decision old enough to resolve, replays subsequent 5-min bars
     (TopStep history API) against the stated stop/target/15:55 flat and fills in
     the hypothetical P&L. Prints the running scoreboard vs a coin-flip.

Verdict rule (agreed): 30+ decisions before ANY conclusion; the analyst must beat
the coin flip AND the mechanical bot's baseline over the same days to earn a role.
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import live_config as cfg
from topstep.auth import get_session_token
from topstep.contracts import CONTRACTS
from topstep.history import get_bars, MINUTE, DAY

ET = ZoneInfo(cfg.TZ)
LOG = ROOT / "logs" / "advisor_decisions.jsonl"

DECISION_SCHEMA = ('{"action":"long|short|stand_aside","stop_pts":<float>,'
                   '"target_pts":<float>,"confidence":<0-100>,"thesis":"<one sentence>"}')


def snapshot():
    token = get_session_token()
    cid = CONTRACTS[cfg.CONTRACT]
    now = datetime.now(timezone.utc)
    bars5 = sorted(get_bars(token, cid, (now - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            now.strftime("%Y-%m-%dT%H:%M:%SZ"), unit=MINUTE, unit_number=5,
                            limit=400, include_partial=False), key=lambda b: b["t"])
    barsD = sorted(get_bars(token, cid, (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            now.strftime("%Y-%m-%dT%H:%M:%SZ"), unit=DAY, unit_number=1,
                            limit=40, include_partial=False), key=lambda b: b["t"])
    if len(bars5) < 50 or len(barsD) < 6:
        raise RuntimeError("not enough data")

    now_et = datetime.now(ET)
    today = now_et.date().isoformat()
    tb = [b for b in bars5 if b["t"][:10] == today]
    last = bars5[-1]
    d_prev = barsD[-2] if barsD[-1]["t"][:10] == today else barsD[-1]
    ranges = [b["h"] - b["l"] for b in barsD[-6:-1]]
    day_hi = max(b["h"] for b in tb) if tb else None
    day_lo = min(b["l"] for b in tb) if tb else None

    return {
        "time_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "price": last["c"],
        "day_high": day_hi, "day_low": day_lo,
        "prior_day_high": d_prev["h"], "prior_day_low": d_prev["l"],
        "prior_close": d_prev["c"],
        "gap_pts": round(tb[0]["o"] - d_prev["c"], 2) if tb else None,
        "last_2h_change": round(last["c"] - bars5[-24]["c"], 2),
        "avg_daily_range_5d": round(float(np.mean(ranges)), 1),
        "prior_day_range": round(d_prev["h"] - d_prev["l"], 1),
    }


def build_prompt(snap):
    return f"""You are the paper-trade analyst for an MES futures experiment. PAPER ONLY.
Market snapshot: {json.dumps(snap)}
Context you must respect: (1) hour-scale momentum is real but tiny; (2) volatility
clusters — prior_day_range vs avg_daily_range_5d tells you the regime; (3) no
intraday signal we tested beats random after costs, so "stand_aside" is always a
respectable answer; (4) force-flat happens 15:55 ET.
Reply with ONLY one JSON object, no prose: {DECISION_SCHEMA}"""


def cmd_ask():
    snap = snapshot()
    prompt = build_prompt(snap)
    rec = {"asked_at": datetime.now(ET).isoformat(), "snapshot": snap,
           "decision": None, "outcome": None}
    cli = shutil.which("claude")
    if cli:
        out = subprocess.run([cli, "-p", prompt], capture_output=True, text=True,
                             timeout=120).stdout.strip()
        try:
            start, end = out.index("{"), out.rindex("}") + 1
            rec["decision"] = json.loads(out[start:end])
        except (ValueError, json.JSONDecodeError):
            rec["decision"] = {"raw": out}
        _append(rec)
        print("decision logged:", json.dumps(rec["decision"]))
    else:
        _append(rec)
        print("claude CLI not installed - paste this prompt into Claude Code, then run:")
        print("  python -m tools.paper_advisor log '<json answer>'")
        print("-" * 70)
        print(prompt)


def cmd_log(payload):
    recs = _read()
    if not recs or recs[-1]["decision"] is not None:
        print("no pending ask - run `ask` first"); return
    recs[-1]["decision"] = json.loads(payload)
    _write(recs)
    print("decision attached to last ask.")


def cmd_score():
    recs = _read()
    token = get_session_token()
    cid = CONTRACTS[cfg.CONTRACT]
    changed = 0
    for r in recs:
        if r["outcome"] is not None or not r.get("decision") or \
           r["decision"].get("action") in (None, "stand_aside") or "raw" in r["decision"]:
            continue
        asked = datetime.fromisoformat(r["asked_at"])
        if datetime.now(ET) - asked < timedelta(hours=6):
            continue   # not resolvable yet
        d = r["decision"]
        side = 1 if d["action"] == "long" else -1
        start = asked.astimezone(timezone.utc)
        bars = sorted(get_bars(token, cid, start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               (start + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               unit=MINUTE, unit_number=5, limit=200,
                               include_partial=False), key=lambda b: b["t"])
        bars = [b for b in bars if b["t"][:10] == asked.date().isoformat()]
        if len(bars) < 2:
            continue
        entry = bars[0]["o"]
        stop = entry - side * float(d.get("stop_pts", 10) or 10)
        tgt = entry + side * float(d.get("target_pts", 20) or 20)
        exit_px, how = bars[-1]["c"], "eod"
        for b in bars:
            t_et = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
            if t_et.hour * 60 + t_et.minute >= cfg.FLAT_MIN:
                exit_px, how = b["o"], "flat1555"; break
            if (side > 0 and b["l"] <= stop) or (side < 0 and b["h"] >= stop):
                exit_px, how = stop, "stop"; break
            if (side > 0 and b["h"] >= tgt) or (side < 0 and b["l"] <= tgt):
                exit_px, how = tgt, "target"; break
        pts = (exit_px - entry) * side
        r["outcome"] = {"entry": entry, "exit": exit_px, "how": how,
                        "pts": round(pts, 2), "usd_1micro": round(pts * 5 - 3.12, 2)}
        changed += 1
    _write(recs)

    scored = [r for r in recs if r["outcome"]]
    aside = sum(1 for r in recs if r.get("decision") and r["decision"].get("action") == "stand_aside")
    print(f"decisions: {len(recs)} | scored: {len(scored)} (+{changed} new) | stand_aside: {aside}")
    if scored:
        pnl = [r["outcome"]["usd_1micro"] for r in scored]
        wins = sum(1 for p in pnl if p > 0)
        print(f"hypothetical 1-micro P&L: total ${sum(pnl):.2f} | avg ${np.mean(pnl):.2f} | "
              f"win {wins}/{len(scored)}")
        print("verdict requires n>=30 AND beating both a coin flip and the bot baseline.")


def _read():
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(recs):
    LOG.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def _append(rec):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"
    if cmd == "ask":
        cmd_ask()
    elif cmd == "log":
        cmd_log(sys.argv[2])
    elif cmd == "score":
        cmd_score()
    else:
        print("usage: python -m tools.paper_advisor [ask|log '<json>'|score]")
