"""
Phase D: Loop every trading day, run strategy, apply costs, record results.
"""
import pandas as pd
from backtest.data import load_cached, group_by_day
from backtest.strategy import run_day, Trade
from backtest.costs import apply_costs
from backtest.config import SLIPPAGE_TICKS


def run_backtest(slippage_ticks: int = SLIPPAGE_TICKS, day_groups=None, strategy_fn=run_day) -> pd.DataFrame:
    if day_groups is None:
        day_groups = group_by_day(load_cached())
    records = []

    for date, day_bars in day_groups:
        trade = strategy_fn(day_bars, date)
        if trade is None:
            continue

        if not trade.skipped:
            trade.net_pnl = apply_costs(trade.gross_pnl, slippage_ticks)

        records.append({
            "date":          trade.date,
            "side":          trade.side,
            "or_width":      trade.or_width,
            "entry_price":   trade.entry_price,
            "exit_t1":       trade.exit_price_t1,
            "exit_t2":       trade.exit_price_t2,
            "exit_reason":   trade.exit_reason,
            "gross_pnl":     trade.gross_pnl,
            "net_pnl":       trade.net_pnl,
            "skipped":       trade.skipped,
            "skip_reason":   trade.skip_reason,
        })

    results = pd.DataFrame(records)
    if not results.empty:
        results["equity_curve"] = results["net_pnl"].cumsum()
    return results
