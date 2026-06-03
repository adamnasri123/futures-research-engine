from datetime import time

# Contract
CONTRACT = "MES"  # key into contracts.CONTRACTS dict

# Session (US Eastern — all strategy logic works in ET)
SESSION_OPEN  = time(9, 30)   # equity cash open — NOT 9:00
SESSION_CLOSE = time(11, 30)  # time-stop: force-close all positions

# Opening Range
OR_MINUTES         = 5        # 9:30:00 – 9:34:59 ET
OR_MIN_WIDTH_PTS   = 3.0      # skip day if OR width < this (index points)
MAX_VWAP_CROSSES   = 3        # skip day if price crosses VWAP this many times in first hour

# --- Trend-ride ("catch the wave") strategy ---
TREND_SWING_K        = 2      # bars each side for a fractal swing point (confirmed K bars later)
TREND_ATR_PERIOD     = 14     # ATR lookback (5-min bars)
TREND_ROOM_ATR_MULT  = 1.0    # "liquidity not near": overhead resistance must be > this * ATR away
TREND_ENTRY_END_MIN  = 12 * 60        # last minute-of-day to OPEN a trade (12:00 ET)
TREND_FLAT_MIN       = 15 * 60 + 55   # force-flat time (15:55 ET)
TREND_MIN_RISK_PTS   = 2.0    # skip if initial stop distance < this (avoids micro-stops)

# --- "Enter sooner, give it room" variant ---
# Tests the finding that the edge is the trend-DAY direction read, not the entry
# trigger: enter immediately on confirmed trend, use a WIDE ATR stop, ride to close.
TREND_IMM_STOP_ATR   = 2.0    # wide protective stop = this * ATR at entry

# Targets & stops (multiples of OR width)
TARGET_1_R         = 1.0      # T1: exit 50% of position
TARGET_2_R         = 2.0      # T2: exit remainder
VWAP_CROSSINGS_WINDOW_MINS = 60

# Cost model
COMMISSION_RT      = 0.62     # round-turn per MES contract (Topstep rate — update when verified)
SLIPPAGE_TICKS     = 1        # ticks of slippage on each entry and exit fill
TICK_SIZE          = 0.25     # MES tick size in index points
TICK_VALUE         = 1.25     # $ per tick for MES

# Stress test
SLIPPAGE_STRESS_TICKS = 2     # re-run with this slippage to see if edge survives

# Contract specs
POINT_VALUE = TICK_VALUE / TICK_SIZE  # $5 per point for MES

# Backtesting split
IN_SAMPLE_FRACTION = 0.70     # first 70% of days = in-sample, last 30% = out-of-sample

# Random benchmark
BENCHMARK_RUNS = 1000

# Topstep account
ACCOUNT_SIZE       = 50_000
TRAILING_MLL       = 2_000    # max loss limit trails from equity peak

# Data cache
CACHE_DIR = "backtest/cache"

# --- Modular parameter sweep ---
SWEEP_EMA_TREND      = 10      # EMA period for trend-bias on the trend timeframe
SWEEP_EMA_EXEC       = 20      # EMA period for exec-TF pullback/trail
SWEEP_ATR_PERIOD     = 14
SWEEP_DONCHIAN_N     = 20      # Donchian breakout lookback
SWEEP_BREAKOUT_N     = 6       # short breakout lookback
SWEEP_MOMENTUM_ATR   = 1.5     # momentum bar range = this * ATR
SWEEP_RETEST_TOL_ATR = 0.25    # retest must come within this * ATR of the level
SWEEP_TARGET_R       = 1.5     # target_trail: take half at this R, trail the rest
SWEEP_ENTRY_END_MIN  = 14 * 60         # last entry time 14:00 ET
SWEEP_FLAT_MIN       = 15 * 60 + 55    # force-flat 15:55 ET
SWEEP_MIN_TRADES     = 30      # ignore combos with fewer trades on a split

# 3-way chronological split for the sweep
SWEEP_TRAIN_FRAC = 0.50
SWEEP_VAL_FRAC   = 0.25
# holdout = remaining 0.25 (never touched during search)
