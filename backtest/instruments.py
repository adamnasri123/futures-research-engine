"""
Per-instrument economics for the multi-instrument sweep.

CRITICAL: P&L and the $2000 MLL check depend on $ value per point, which differs
hugely by instrument. Using the wrong value silently corrupts every result. We trade
the MICRO of each (eval-appropriate, smallest size) and use its point value.

point_value = $ per 1.0 index/price point for the MICRO contract.
tick_size   = minimum price increment.
(commission ~ same micro ballpark; we reuse the MES round-turn as an approximation —
flagged: real per-instrument commissions vary slightly but are all sub-$1 round-turn.)

Sources: CME contract specs (micro/e-mini), standard as of 2025-26.
"""

# name -> dict(micro_symbol, point_value, tick_size)
INSTRUMENTS = {
    # Equity index micros
    "NQ":  {"micro": "MNQ", "point_value": 2.00,  "tick_size": 0.25},   # Micro Nasdaq: $2/pt
    "YM":  {"micro": "MYM", "point_value": 0.50,  "tick_size": 1.0},    # Micro Dow: $0.50/pt
    "RTY": {"micro": "M2K", "point_value": 5.00,  "tick_size": 0.10},   # Micro Russell: $5/pt
    # Bonds (no true "micro"; use full but tiny tick value — flagged)
    "ZB":  {"micro": "ZB",  "point_value": 1000.0, "tick_size": 1/32}, # T-Bond: $1000/pt (full)
    "ZN":  {"micro": "ZN",  "point_value": 1000.0, "tick_size": 1/64}, # 10Y note: $1000/pt (full)
    # Metals micros
    "GC":  {"micro": "MGC", "point_value": 10.0,  "tick_size": 0.10},   # Micro Gold: $10/pt
    "SI":  {"micro": "SIL", "point_value": 1000.0,"tick_size": 0.005},  # Micro Silver: $1000/pt (1000oz? -> see note)
    # Energy micro
    "CL":  {"micro": "MCL", "point_value": 100.0, "tick_size": 0.01},   # Micro Crude: $100/pt
    # Control
    "ES":  {"micro": "MES", "point_value": 5.00,  "tick_size": 0.25},   # Micro S&P: $5/pt
}

# NOTE/RISK FLAGS (do not trust blindly):
#  - ZB/ZN: full-size contracts ($1000/pt). A single point move = $1000 -> with a $2000
#    MLL these are NOT eval-appropriate at all; included for research completeness but
#    flag any "GO" as needing a true micro (none exists for these).
#  - SI/SIL: silver micro economics are unusual; SIL (1000oz) ~ $5/cent. point_value here
#    is approximate -> treat SI results as INDICATIVE, re-verify specs before trust.
#  - Equity micros (MNQ/MYM/M2K/MES) and MGC/MCL are the clean, eval-appropriate ones.
EVAL_CLEAN = {"NQ", "YM", "RTY", "GC", "CL", "ES"}   # trust these most
EVAL_FLAGGED = {"ZB", "ZN", "SI"}                     # economics caveats
