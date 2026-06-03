"""
Phase E: Performance stats + TopStep survival checks.
"""
import pandas as pd
from backtest.config import ACCOUNT_SIZE, TRAILING_MLL, TICK_VALUE


def compute_metrics(results: pd.DataFrame, label: str = "") -> dict:
    trades = results[~results["skipped"]].copy()
    n = len(trades)

    if n == 0:
        print(f"\n[{label}] No trades to analyse.")
        return {}

    wins   = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    win_rate   = len(wins) / n
    avg_win    = wins["net_pnl"].mean()    if len(wins)   > 0 else 0
    avg_loss   = losses["net_pnl"].mean()  if len(losses) > 0 else 0
    payoff     = abs(avg_win / avg_loss)   if avg_loss != 0 else float("inf")
    gross_wins = trades[trades["gross_pnl"] > 0]["gross_pnl"].sum()
    gross_loss = abs(trades[trades["gross_pnl"] <= 0]["gross_pnl"].sum())
    pf         = gross_wins / gross_loss   if gross_loss > 0 else float("inf")
    expectancy = trades["net_pnl"].mean()

    # Max drawdown
    equity = trades["net_pnl"].cumsum()
    peak   = equity.cummax()
    dd     = equity - peak
    max_dd = dd.min()

    # Max consecutive losers
    streak = max_consec_losses(trades["net_pnl"].tolist())

    # TopStep trailing MLL simulation
    mll_breach, mll_equity = simulate_mll(trades["net_pnl"].tolist())

    metrics = {
        "label":            label,
        "n_trades":         n,
        "n_skipped":        results["skipped"].sum(),
        "win_rate":         win_rate,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "payoff_ratio":     payoff,
        "profit_factor":    pf,
        "total_net_pnl":    trades["net_pnl"].sum(),
        "expectancy":       expectancy,
        "max_drawdown":     max_dd,
        "max_consec_loss":  streak,
        "mll_breach":       mll_breach,
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(m: dict) -> None:
    label = f" [{m['label']}]" if m["label"] else ""
    print(f"\n{'='*55}")
    print(f"  RESULTS{label}")
    print(f"{'='*55}")
    print(f"  Trades analysed    : {m['n_trades']}  (skipped: {m['n_skipped']})")

    if m["n_trades"] < 100:
        print(f"  *** WARNING: fewer than 100 trades — results not statistically reliable ***")

    print(f"  Win rate           : {m['win_rate']:.1%}")
    print(f"  Avg win            : ${m['avg_win']:.2f}")
    print(f"  Avg loss           : ${m['avg_loss']:.2f}")
    print(f"  Payoff ratio       : {m['payoff_ratio']:.2f}x")
    print(f"  Profit factor      : {m['profit_factor']:.2f}")
    print(f"  Expectancy/trade   : ${m['expectancy']:.2f}")
    print(f"  Total net P&L      : ${m['total_net_pnl']:.2f}")
    print(f"  Max drawdown       : ${m['max_drawdown']:.2f}")
    print(f"  Max consec losses  : {m['max_consec_loss']}")
    print(f"  TopStep MLL breach : {'YES — UNUSABLE at this size' if m['mll_breach'] else 'No'}")
    print(f"{'='*55}")


def max_consec_losses(pnl_list: list) -> int:
    best = cur = 0
    for p in pnl_list:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)
    return best


def simulate_mll(pnl_list: list) -> tuple[bool, list]:
    """Simulate TopStep $2,000 trailing MLL against the equity curve."""
    equity = 0.0
    peak   = 0.0
    mll    = -TRAILING_MLL   # starts at -2000 from starting peak (0)
    curve  = []
    breach = False

    for pnl in pnl_list:
        equity += pnl
        if equity > peak:
            peak = equity
            mll  = peak - TRAILING_MLL
        curve.append(equity)
        if equity < mll:
            breach = True
            break

    return breach, curve
