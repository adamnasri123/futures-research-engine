# Strategy Spec & Exit-Check — TopStep Eval

_Last updated: 2026-05-31 (for Monday 2026-06-01 open)_

## ⚠️ Honest status first

Your exit-check was: **"a written strategy spec + backtest results that survived
out-of-sample and random benchmarking + a coded guardrail checklist. If the
backtest failed, you loop back — that's success, not failure."**

**The backtest did NOT survive.** By your own criteria we are in the *loop-back*
state, not the *deploy-with-confidence* state. We are proceeding into the eval
anyway **to build process and discipline on a paper/eval account — not because we
found an edge.** Read that twice. The guardrails below, not the entry signal, are
what protect you.

What we proved across ~5 sessions of testing:
- 360-combo sweep (entries × exits × timeframes × stops): no combo beat random
  entry out-of-sample.
- Regime filtering (trade only trend days) and news filtering (skip FOMC/NFP)
  **do** add value to *day selection* (walk-forward 0.87 → 1.20).
- But even on the best filtered days, intraday entry/exit timing **still did not
  beat a random entry** (best holdout PF 1.33 vs random p95 $681 > our $431).

**Conclusion:** day-selection has weak signal; trade timing has none. We trade
small, rarely, and let the risk caps do the real work.

---

## The plan (what we actually do each day)

### 1. Should we trade today? (`python daily_plan.py`)
Trade **only** if BOTH:
- Daily regime = **TREND** (ADX > 25 or Choppiness < 38.2, and not choppy), AND
- **No high-impact news** (not an FOMC or NFP day).

Otherwise: **stand aside.** Most days we will not trade. That is correct.

### 2. If trading — execution rules (discipline, not prediction)
| Rule | Value |
|---|---|
| Instrument | 1 micro contract (see open question — MES vs MNQ) |
| Timeframe | 5-min bars |
| Bias | long if price > 10-EMA & EMA rising; short if mirror |
| Entry | break of last 6-bar high/low in bias direction |
| Stop | 2.5 × ATR(14) from entry — **always bracketed** |
| Target | bracket take-profit at 2R (configurable) |
| Exit | stop, target, close back through 20-EMA, or force-flat |
| Entry window | 9:35 – 12:00 ET only |
| Force flat | by cutoff (see open question), no exceptions |

### 3. Risk caps (HARD — enforced in code)
| Cap | Value | Enforced by |
|---|---|---|
| Max trades/day | 2 | `live_guard.preflight` |
| One-trade-per-day flag | once it trades, done | `preflight` (open-position + count) |
| Daily loss limit | stop at −$500 net | `preflight` blocks new entries near cap |
| Daily profit cap | stop at +$1500 net | `preflight` (consistency rule) |
| Position size | 1 micro, period | `place_bracketed` refuses size ≠ 1 |
| Bracket every entry | no naked orders, ever | `place_bracketed` raises if no stop+target |
| Force flat at cutoff | auto-close | `force_flat_if_needed` |
| Trailing drawdown | size so a normal red day can't breach $2000 | 1 micro × 2 trades ≪ $2000 |

---

## Coded guardrail checklist (verify before live)
- [x] `place_bracketed()` raises if stop or target missing (no naked orders)
- [x] `place_bracketed()` raises if size ≠ 1 micro
- [x] `preflight()` blocks when near daily loss cap
- [x] `preflight()` blocks at daily profit cap
- [x] `preflight()` blocks after max trades / open position (one-and-done)
- [x] `preflight()` blocks outside the entry window
- [x] `force_flat_if_needed()` closes any open position at the cutoff
- [x] `daily_plan.py` gives the trade/stand-aside call before the open
- [x] `check_status.py` shows live P&L vs caps intraday
- [ ] **Decision pending:** auto-execute vs decision-support (see below)
- [ ] **Decision pending:** contract (MES vs MNQ) and force-flat time
- [ ] Dry-run on Monday in DECISION-SUPPORT mode before any auto-execution

---

## Daily routine
1. **Before 9:30 ET** — `python daily_plan.py` → trade or stand aside.
2. **If trading** — wait for the entry window (9:35+), follow the rules, place the
   trade **as a bracket** (entry + stop + target together).
3. **During session** — `python check_status.py` to watch P&L vs caps.
4. **By the cutoff** — `python -c "from live_guard import *; from auth import get_session_token; from accounts import get_accounts; t=get_session_token(); a=get_accounts(t)[0]['id']; print(force_flat_if_needed(t,a))"` (or the bot does it) → flat.
5. **End of day** — note result; once per day, done.

---

## Loop-back triggers (when to STOP and rethink)
- Two consecutive losing days, or
- Any single day that breaches a cap the code didn't catch, or
- Live results diverging hard from the (already weak) backtest.

If any fires: stop trading, review, and treat it as success — the system caught a
problem before it cost real money.
