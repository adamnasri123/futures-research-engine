# Futures Trading Research & Execution Engine

A from-scratch quantitative research framework for intraday futures strategies, built
around one principle: **refuse to fool yourself.** It pairs a rigorous, overfitting-resistant
backtesting engine with a live execution harness for the [TopstepX / ProjectX](https://gateway.docs.projectx.com/)
futures API.

## The headline finding

I used this engine to test **12+ strategy families across 9 futures instruments** (ES, NQ,
YM, RTY, GC, CL, SI, ZB, ZN) — opening-range breakouts, VWAP, trend-following, market-structure
"liquidity sweeps," regime filtering, news filtering, multi-timeframe, and position-sizing
schemes. Every approach was put through out-of-sample validation, walk-forward analysis, a
random-entry benchmark, and a multiple-testing correction (White's Reality Check).

**Conclusion: none produced an edge that survived honest validation.** Simple price-based
intraday signals on liquid futures did not beat a random-entry benchmark once costs and
out-of-sample testing were applied.

That is the point of the project. The engineering value here is not a magic strategy — it's a
system disciplined enough to *prove a strategy doesn't work* before risking capital on it,
which is exactly the failure mode that ruins most retail traders.

## Why this is hard (and what it demonstrates)

A backtest that ignores these will lie to you. This engine handles all of them:

- **Look-ahead bias** — signals are detected on a bar's close and filled on the *next* bar's
  open; higher-timeframe context only uses bars that have actually closed.
- **Overfitting** — chronological train / validation / holdout splits; the holdout is touched
  exactly once. Plus independent walk-forward (rolling re-optimization on unseen windows).
- **Multiple testing** — testing N strategies guarantees a lucky winner. The acceptance bar
  scales with the search size (Reality Check): a survivor must beat the 95th percentile of the
  *best-of-N* random strategies. Testing 360 combos raises the bar to what luck alone produces.
- **The "is it just beta?" trap** — a direction-matched random-entry benchmark isolates whether
  entry *timing* adds value or the strategy is just riding market drift.
- **Realistic costs** — per-instrument commission + slippage, with a stress test at 2× slippage.
- **Prop-firm risk** — simulates TopStep's $2,000 trailing drawdown to check whether a strategy
  is even survivable at a given position size, independent of its raw P&L.

## Architecture

```
├── auth / accounts / contracts / orders / positions / history / trades   # ProjectX API client
├── live_config.py      # all live parameters + hard risk caps
├── live_guard.py       # enforced guardrails: no naked orders, daily loss cap,
│                       #   one-trade-per-day, position-size ceiling, force-flat
├── autotrader.py       # autonomous live runtime (regime/news day-filter + execution)
├── daily_plan.py       # pre-market trade / stand-aside decision
└── backtest/
    ├── data.py          # market-data pipeline (quarterly contract stitching, RTH + 24h)
    ├── strategy*.py     # strategy implementations (ORB, VWAP, trend, modular grid)
    ├── engine.py        # event loop, cost model, metrics
    ├── metrics.py       # PF, expectancy, drawdown, trailing-MLL simulation
    ├── benchmark.py     # random-entry + direction-matched benchmarks
    ├── sweep.py         # train/val/holdout grid sweep + walk-forward
    ├── regime.py/news.py # ADX/Choppiness regime classifier; economic-event calendar
    ├── liquidity.py / overnight.py  # market-structure / liquidity-pool strategies
    └── multisweep.py    # cross-instrument sweep with per-instrument economics
```

## The honest-validation gates (a strategy ships only if it clears ALL)

1. 100+ trades in sample
2. Out-of-sample profit factor > 1.3
3. Out-of-sample not dramatically worse than in-sample
4. Beats the random-entry benchmark after costs
5. Never breaches the $2,000 trailing drawdown at the chosen size
6. Survives a 2× slippage stress test

## Tech

Python · pandas · NumPy · pyarrow · REST + (planned) SignalR WebSocket for real-time data.

## Notes

- API keys, live trading logs (account numbers, real P&L), and cached third-party market data
  are excluded from version control by design.
- This is a personal research project, not financial advice. The live execution layer runs on a
  prop-firm evaluation account; the documented conclusion is that the tested signals have no
  proven edge, and the risk caps — not the signal — are what protect the account.
