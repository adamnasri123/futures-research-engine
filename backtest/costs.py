"""
Phase C: Realistic cost model.
Commission + slippage applied to every fill.
"""
from backtest.config import COMMISSION_RT, SLIPPAGE_TICKS, TICK_SIZE, TICK_VALUE, POINT_VALUE


def apply_costs(gross_pnl: float, slippage_ticks: int = SLIPPAGE_TICKS) -> float:
    """
    Deduct round-turn commission and slippage (entry + exit) from gross P&L.
    Slippage applied on BOTH entry and exit fills.
    """
    slippage_per_fill = slippage_ticks * TICK_VALUE
    total_slippage    = slippage_per_fill * 2   # entry + exit
    total_cost        = COMMISSION_RT + total_slippage
    return gross_pnl - total_cost
