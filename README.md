# Futures Trading Research & Execution Engine

A from-scratch quantitative research framework for intraday futures strategies, built
around one principle: **refuse to fool yourself.** It pairs a rigorous, overfitting-resistant
backtesting engine with a live execution harness and dashboard for the
[TopstepX / ProjectX](https://gateway.docs.projectx.com/) futures API.

## The headline finding

This engine tested **12+ strategy families across 9 futures instruments** (ES, NQ, YM, RTY,
GC, CL, SI, ZB, ZN) — opening-range breakouts, VWAP, trend-following, market-structure
"liquidity sweeps," regime filtering, news/event reactions, multi-timeframe, and
position-sizing schemes. Every approach went through out-of-sample validation, walk-forward
analysis, a random-entry benchmark, and a multiple-testing correction (White's Reality Check).

**Conclusion: none produced an edge that survived honest validation.** Simple price-based
intraday signals on liquid futures did not beat a random-entry benchmark once costs and
out-of-sample testing were applied.

That is the point of the project. The engineering value is not a magic strategy — it's a
system disciplined enough to *prove a strategy doesn't work* before risking capital on it,
which is exactly the failure mode that ruins most retail traders.

A second, harder-won lesson (2026-06 audit): **a backtest must replicate the live system's
data window, not just its rules.** The live bot computed indicators over 8 hours of bars
(including overnight); the original replays warmed up on regular-session bars only. Same
rules, different entry distribution — different strategy. `backtest/live_replay.py` now
mirrors the live path bar-for-bar and is validated by a parity test against real live
trades (5/6 exact matches on side, timing, and stop size).

## Repository layout

```
├── autotrader.py        # autonomous live runtime (day filter + execution + kill switch)
├── live_config.py       # all live parameters + hard risk caps (the contract with yourself)
├── live_guard.py        # enforced guardrails: no naked orders, daily loss cap,
│                        #   trade limit, size ceiling, post-entry stop verification, force-flat
├── daily_plan.py        # pre-market trade / stand-aside decision (human-readable)
├── check_status.py      # intraday P&L vs caps from the command line
├── run_daily.ps1        # Task Scheduler launcher
├── topstep/             # ProjectX API client (auth, accounts, contracts, orders,
│                        #   positions, trades, history)
├── dashboard/           # local web dashboard (live trades, P&L, MLL gauge, STOP button)
├── backtest/            # research engine
│   ├── data.py            # market-data pipeline (contract stitching, RTH + 24h caches)
│   ├── engine.py / metrics.py / costs.py / benchmark.py
│   ├── live_replay.py     # live-faithful replay + parity test (start here)
│   ├── strategy*.py / modular.py / sweep.py / multisweep.py
│   ├── regime.py / news.py / news_event.py
│   └── liquidity.py / overnight.py / sizing*.py / goal_sizing.py / entry_timing.py
├── docs/                # STRATEGY.md, BACKTEST_REPORT.md, sweep result artifacts
└── logs/                # bot logs + analyst journal (gitignored — real P&L)
```

## Quick start — connecting to your TopStep account

1. **Prerequisites:** Windows (PowerShell), Python 3.13+, a funded or evaluation
   [Topstep](https://www.topstep.com/) account on the TopstepX platform.

2. **Get API access:** in TopstepX go to **Settings → API Access**, subscribe to API
   access, and generate a **personal API key**. Note your TopstepX **username**.

3. **Install:**
   ```powershell
   git clone <this-repo> topstep-bot ; cd topstep-bot
   python -m venv venv
   .\venv\Scripts\pip install -r requirements.txt
   ```

4. **Credentials:** copy `.env.example` to `.env` and fill in `TOPSTEP_USERNAME` and
   `TOPSTEP_API_KEY` (and `MASSIVE_API_KEY` only if you want to rebuild backtest data).

5. **Verify the connection (read-only):**
   ```powershell
   .\venv\Scripts\python -m topstep.auth        # should print a session token
   .\venv\Scripts\python check_status.py        # account, balance, today's P&L
   ```

6. **CRITICAL — platform setting:** in TopstepX order settings enable **Auto OCO**
   (bracket) orders. Without it the API rejects attached stop/target brackets and the
   bot will refuse to trade (it never sends naked orders).

7. **Dry-run the bot** (no orders placed; full decision pipeline + logs):
   ```powershell
   .\venv\Scripts\python autotrader.py          # DRY-RUN is the default
   ```
   Read `logs/autotrader_YYYY-MM-DD.log`. Run several clean dry days before going live.

8. **Go live (your decision, your risk):** edit `run_daily.ps1` (the `--live` flag) and
   register it with Task Scheduler to fire each weekday at 9:25 AM ET:
   ```powershell
   schtasks /Create /TN "trade" /TR "powershell -File C:\path\to\run_daily.ps1" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:25
   ```

9. **Dashboard** (optional, recommended):
   ```powershell
   .\dashboard\run_dashboard.ps1                # opens http://127.0.0.1:8765
   ```
   Shows balance, trailing-drawdown cushion, today's fills with prices/times, open
   position and working orders, the live bot log, a **STOP BOT** kill switch (flattens
   and halts the bot within ~45 s), and a one-click **analyst briefing** you can paste
   into Claude Code for a structured review session.

## The honest-validation gates (a strategy ships only if it clears ALL)

1. 100+ trades in sample
2. Out-of-sample profit factor > 1.3
3. Out-of-sample not dramatically worse than in-sample
4. Beats the random-entry benchmark after costs
5. Never breaches the $2,000 trailing drawdown at the chosen size
6. Survives a 2× slippage stress test

…and since the audit: **7. The backtest must pass a parity test against the live
system's actual fills** before its numbers are believed.

## Why this is hard (what the engine defends against)

- **Look-ahead bias** — signals on bar close, fills on next bar open; swing points only
  count after their confirmation bar; regime labels use prior-day data only.
- **Overfitting** — chronological train/validation/holdout; the holdout is touched once.
- **Multiple testing** — the acceptance bar scales with the number of strategies tried
  (best-of-N random benchmark / Reality Check).
- **Backtest-vs-live drift** — the live-faithful replay + parity test (the 2026-06 audit
  found the original replays sampled a different entry window than the live bot).
- **Realistic costs** — commission + slippage on every fill, 2× slippage stress test.
- **Prop-firm risk** — simulates the $2,000 trailing max-loss limit at the chosen size.

## Safety model (live layer)

- The bot is a **deterministic script** — no AI, no discretion at runtime, one entry
  point per day, hard force-flat at 15:55 ET.
- `live_guard.py` enforces: bracket-or-nothing orders, post-entry server-side stop
  verification (flattens immediately if no protective stop is confirmed), daily loss/profit
  caps, max trades/day, contract-size ceiling, weekend and early-close-day stand-downs,
  FOMC/NFP day filter, stale-data guard, and a file-based kill switch.
- Risk caps — not the entry signal — are the protection. The research says the signal
  has no proven edge; the caps are what keep a no-edge system alive.

## Tech

Python · pandas · NumPy · pyarrow · Flask (dashboard) · REST (ProjectX + Massive.com).

## Notes

- API keys, live trading logs (account numbers, real P&L), and cached third-party market
  data are excluded from version control by design.
- This is a personal research project, **not financial advice**. The documented conclusion
  is that the tested signals have no proven edge; anyone running the live layer is trading
  market drift inside risk caps, and should size accordingly (the backtests say: 1 micro).
