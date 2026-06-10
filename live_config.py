"""
Live trading configuration — the rules the bot/operator follows during the eval.

IMPORTANT HONESTY NOTE: backtesting showed the intraday ENTRY/EXIT rules below do
NOT beat random entry. The parts with real evidence are (a) day-selection (only
trade trending, non-news days) and (b) the hard risk caps. Treat the risk caps as
the primary protection, not the signal.
"""

# --- Instrument (MES micro S&P) ---
CONTRACT      = "MES"          # key into contracts.CONTRACTS
TICK_SIZE     = 0.25
TICK_VALUE    = 1.25           # $ per tick
POINT_VALUE   = 5.0           # $ per index point (4 ticks)

# --- Risk caps (eval phase — hard stops for the day) ---
DAILY_LOSS_CAP    = 500.0     # stop trading for the day at -$500 realized
DAILY_PROFIT_CAP  = 1500.0    # stop trading for the day at +$1500 (consistency rule)
MAX_TRADES_PER_DAY = 2        # 2 trades max
CONTRACTS_PER_TRADE = 5       # user-chosen 5 micros (HIGH accepted eval risk — eyes open).
                              # 2026-06-09 faithful-replay numbers (best config, 10:45+):
                              # worst single trade -$1891 ≈ one fill from MLL breach; a
                              # NORMAL losing cluster breaches the $2000 trailing MLL by
                              # ~OOS trade #6-8 at this size. 1 micro is the only size
                              # that survives historically. User informed, chose 5. See
                              # ANALYST_JOURNAL 2026-06-09 + logs/RESEARCH_LOG.md.
MAX_CONTRACTS_HARD = 5        # absolute ceiling — code must never place more than this

# TopStep 50k eval account
ACCOUNT_TRAILING_MLL = 2000.0  # never let equity fall $2000 from peak

# --- Day-selection filter (the part with evidence) ---
# Only trade days the daily regime flags TREND and that are NOT high-impact news days.
REGIME_LOOKBACK_DAYS = 90      # daily bars to pull for ADX/Choppiness
REGIME_MIN_BARS      = 28      # need >= this many daily bars for a valid reading

# --- Execution rules (mechanical; no proven edge — discipline only) ---
EXEC_TF_MIN     = 5           # trade on 5-min bars
EMA_TREND       = 10          # intraday bias EMA (long if price>EMA & EMA rising)
BREAKOUT_N      = 6           # enter on break of last N-bar high/low in bias direction
ATR_PERIOD      = 14
STOP_ATR        = 2.5         # protective stop = this * ATR at entry
# ENTRY WINDOW (changed 2026-06-09 after the live-replay audit): entries used to start
# at 9:35, but with the bot's 8h data window the signal fired on the open gap nearly
# every day — a configuration that was never backtested and that the faithful replay
# (backtest/live_replay.py) shows is NEGATIVE out-of-sample (PF 0.58 on TREND days).
# Entries >= 10:45 are the variant the original validation numbers actually describe
# (OOS PF 1.15 all-days / +$1056-1104, drift-level, NOT proven edge vs random).
ENTRY_START_MIN = 10 * 60 + 45   # earliest entry 10:45 ET
ENTRY_END_MIN   = 12 * 60        # no NEW trades after 12:00 ET (morning focus)
FLAT_MIN        = 15 * 60 + 55   # force flat by 15:55 ET

# CME early-close days (equity futures close ~13:00 ET): the 15:55 force-flat can
# never run, so the bot STANDS ASIDE entirely. Re-verify each year (cmegroup.com).
EARLY_CLOSE_DAYS = {
    "2026-07-03",   # Independence Day (observed)
    "2026-11-26",   # Thanksgiving
    "2026-11-27",   # day after Thanksgiving
    "2026-12-24",   # Christmas Eve
}

# Dashboard / operator kill switch: if this file exists in the project root, the bot
# flattens (if live) and exits at the next poll. Created by the dashboard STOP button.
STOP_FLAG_FILE = "STOP_BOT"

# Single-instance heartbeat: the bot touches this file every poll. A fresh heartbeat
# makes a second launch refuse to start, and tells the dashboard the bot is alive.
HEARTBEAT_FILE = "logs/.bot_heartbeat"
HEARTBEAT_FRESH_SEC = 120

# Session timezone
TZ = "America/New_York"
