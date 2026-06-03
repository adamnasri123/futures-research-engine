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
CONTRACTS_PER_TRADE = 5       # user-chosen 5 micros (HIGH accepted eval risk — eyes open):
                              # backtest shows the worst SINGLE trade = -$2141 at 5 micros,
                              # which breaches the $2000 trailing MLL in ONE fill; worst
                              # drawdown breaches by ~trade #17; only ~5 consecutive losses
                              # wipe $2000 (worst historical streak was 10+). See ANALYST_JOURNAL.
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
EMA_EXIT        = 20          # exit when price closes back through this EMA
ENTRY_START_MIN = 9 * 60 + 35   # earliest entry 9:35 ET (let the open settle one bar)
ENTRY_END_MIN   = 12 * 60        # no NEW trades after 12:00 ET (morning focus)
FLAT_MIN        = 15 * 60 + 55   # force flat by 15:55 ET

# Session timezone
TZ = "America/New_York"
