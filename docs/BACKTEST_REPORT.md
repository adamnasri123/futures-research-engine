# Backtest Report — Intraday Futures Strategy Research

_Generated 2026-06-03. Source: recorded results in the analyst journal (the `.txt`
result artifacts were truncated; all figures below are from when each test ran)._

> **2026-06-09 AUDIT ADDENDUM — read before using §4 or any "live strategy replay"
> figure.** A code audit found that every replay of the live strategy ran on RTH-only
> data with in-day indicator warm-up, so it could only enter from ~10:45 ET — while the
> live bot (8-hour data window incl. overnight) enters at 9:35 on 80% of days. A new
> live-faithful replay (`backtest/live_replay.py`, parity-validated against the real
> June trades 5/6 exactly) shows the **as-live config is NEGATIVE out-of-sample: PF
> 0.58, −$25/trade at 1 micro, MLL breach by ~trade #6 at 5 micros** — at or below the
> random-entry benchmark. Restricting entries to ≥10:45 reproduces the +$1,056 OOS
> shown below, i.e. §4's numbers are correct *for that variant only*. The overall
> NO-GO conclusions of this report are unchanged (they get stronger). Details:
> `logs/RESEARCH_LOG.md`. Also fixed: FOMC 2026 calendar was missing all dates after
> April (incl. June 17, 2026).

## 1. Summary

A Python backtesting framework was used to test whether simple, mechanical intraday
futures strategies have a **statistically robust edge** that survives honest validation
and realistic costs. Across **12+ strategy families on 9 instruments**, the answer was
consistently **no**: nothing beat a random-entry benchmark out-of-sample after a
multiple-testing correction.

**Headline finding:** entry/exit *timing* has no edge over random on liquid futures.
Day-selection (trade only trending, non-news days) has a weak signal. The risk caps,
not the signal, are what protect the account.

## 2. Method (how each test was judged)

Every strategy had to clear the same honest gauntlet:

| Check | What it does |
|---|---|
| **Out-of-sample holdout** | Tune on the first ~50–70%, test once on unseen final 25–30%. |
| **Walk-forward analysis** | Re-optimize on a rolling window, trade the next unseen window, repeat. |
| **Random-entry benchmark** | Same exits/costs, random entry. The strategy must beat it. |
| **Multiple-testing (Reality Check)** | Bar = 95th pct of the *best-of-K* random strategies. More combos tried → higher bar. |
| **Cost model** | Commission + slippage on every fill. |
| **Prop-firm risk** | $2,000 trailing max-loss-limit (MLL) breach = eval failure. |

- **Data:** ES (and 8 others) 5-min bars, ~507–514 trading days each (~2 years), via
  Massive.com. RTH cache (9:30–15:55 ET) + a full 24-hour cache (incl. overnight Globex).
- **Instrument:** MES micro (and per-instrument micro economics for the multi-market test).

## 3. Results by strategy

### 3.1 Single-strategy tests (ES)
| Strategy | Key result | Verdict |
|---|---|---|
| 5-min ORB + VWAP | Loses after costs | NO-GO |
| Trend / wave-rider (long only) | Made money = market beta; random longs matched it | NO-GO |
| Wave-rider (long + short) | Random entry made **3.4×** more | NO-GO |
| "Enter sooner + wide stop" | +$1,575 in-sample → **−$1,512 out-of-sample** (textbook overfit) | NO-GO |

### 3.2 Broad mechanical sweep (ES)
- **360 combinations** (6 timeframe-pairs × 6 entries × 5 exits × 2 stops), 50/25/25 split
  + walk-forward.
- Only **3 of 360** survived validation; all failed the holdout.
- Walk-forward stitched **PF 0.92** (losing). **NO-GO.**

### 3.3 Regime + News filtering (ES)
| Filter | Walk-forward PF | Verdict |
|---|---|---|
| Baseline (trade every day) | 0.87 | NO-GO |
| **Trend days only** | **1.20** | NO-GO (best, but still loses to random) |
| Chop days only | 0.71 | NO-GO (sanity check — worst, as expected) |
| Skip FOMC/NFP | 0.83 | NO-GO |
| Trend + skip-news | holdout PF 1.33 | NO-GO (beat zero, lost to random p95 $681) |
- **Finding:** day-selection (trend vs chop, avoiding news) genuinely helps — but even on
  the best filtered days, intraday timing still doesn't beat random.

### 3.4 Liquidity sweeps
- **RTH pools** (60 combos): liquidity-sweep entries *underperformed* a plain breakout
  control. Best survivor +$1,056 vs **Reality-Check bar $3,384** (best-of-60 random). NO-GO.
- **Overnight pools** (24h data, ICT "judas swing", 60 combos): best train PF only 1.12;
  **0 survived validation.** NO-GO. (This was the strongest remaining idea, tested on the
  right data.)

### 3.5 Multi-instrument sweep (the broadest test)
- **8 instruments × 48 combos = 384 strategies**, correct per-micro economics, cross-search
  Reality Check.

| Instrument | Class | Train top PF | Survived validation |
|---|---|---|---|
| NQ (Nasdaq) | equity | 1.25 | 0 |
| YM (Dow) | equity | **1.63** | 0 (best train, total OOS collapse) |
| RTY (Russell) | equity | 1.12 | 1 → failed holdout |
| GC (Gold) | metal | 0.98 | 0 |
| CL (Crude) | energy | 0.90 | 0 |
| SI (Silver) | metal | 0.85 | 0 |
| ZB (T-Bond) | rates | 0.77 | 0 |
| ZN (10Y) | rates | 0.91 | 0 |

- Non-equity markets did **worse**, not better (losing even in-sample).
- One survivor (RTY, +$724 holdout) vs **Reality-Check bar $38,136** — nowhere close.
- **NO-GO across all 8.** Not an ES quirk — it's the nature of liquid futures + retail
  price patterns.

### 3.6 Scenario-adaptive (most promising lead)
- Default breakout + bounce-play at higher-timeframe support/resistance (prior 10-day
  levels, causal).
- Hand-picked params beat baseline OOS (+$1,263 vs +$646) — **but** deeper max drawdown
  (−$1,917) and breached on the full period. A param sweep (64 combos) then showed
  **0 of 64 survived validation.** Overfit. NO-GO.

## 4. Position sizing (does scaling up help?)

Sizing applied to the same trade sequence (it can't change which trades occur, only the
multiplier). Decisive metric = **does it breach the $2,000 MLL** = eval failure.

| Micros | OOS Net | Max drawdown | Eval outcome |
|---|---|---|---|
| **1** | +$1,056 | −$1,231 | **Survives** ($769 cushion) |
| 2 | +$2,113 | −$2,462 | Breach @ trade #66 |
| 3 | +$3,169 | −$3,692 | Breach @ trade #55 |
| 4 | +$4,226 | −$4,923 | Breach @ trade #51 |
| 5 | +$5,282 | −$6,154 | Breach @ trade #17 |

- Win rate (47%) and PF (1.19) are **identical at every size** — sizing only multiplies the
  drawdown. **1 micro is the largest size whose worst losing cluster (10+ trades) fits under
  $2,000.**
- Volatility/$200 sizing, anti-martingale, regime-sizing: all **breach the MLL** out-of-sample.
- Worst single 5-micro trade = **−$2,141** → can breach in one fill.

## 5. Conclusions

1. **No simple price-based intraday strategy showed a validated edge** — on ES or any of 8
   other futures, on RTH or 24-hour data. A dozen+ independent approaches all failed the
   same honest test: strong evidence the edge isn't there, not that it wasn't found.
2. **Day-selection has a weak real signal** (trend > chop, avoid news); intraday *timing*
   does not beat random.
3. **Sizing cannot create edge** — on a no-edge signal it only amplifies the path to ruin.
   Flat 1 micro is the only config that survives the eval drawdown limit.
4. **The validation methodology is the deliverable.** It repeatedly caught false winners
   (overfit in-sample results, luck-driven sweep "winners") before they could cost money —
   which is exactly what a research engine should do.

Any genuine edge, if it exists, would require **new information** beyond historical price
bars: live order-flow / depth-of-market, cross-asset signals, or a different timeframe —
each a separate, uncertain project.
