"""
EXACT sizing boundary: flat 1/2/3/4 micros on the proven baseline (fixed 2:1),
showing net, max drawdown, and the trade# where the $2000 trailing MLL is breached.
Answers 'can we go 2-4 micros?' with the real failure point, not opinion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from backtest.data import load_cached, group_by_day
from backtest.sizing import replay_live_trades, _net_per_contract
from backtest.config import IN_SAMPLE_FRACTION, TRAILING_MLL


def diag(nets):
    eq=0.0; peak=0.0; mll=-TRAILING_MLL; breach=None
    for k,p in enumerate(nets):
        eq+=p
        if eq>peak: peak=eq; mll=peak-TRAILING_MLL
        if breach is None and eq<mll: breach=k+1
    a=np.asarray(nets,float)
    ddc=np.cumsum(a); dd=float((ddc-np.maximum.accumulate(ddc)).min())
    wins=a[a>0]; gl=-a[a<=0].sum(); pf=(wins.sum()/gl) if gl>0 else float("inf")
    return {"net":float(a.sum()),"maxdd":dd,"pf":pf,"breach":breach,"wr":len(wins)/len(a)}


def main():
    groups=group_by_day(load_cached())
    trades=replay_live_trades(groups)
    per1=[_net_per_contract(t["pts"]) for t in trades]
    cut=int(len(per1)*IN_SAMPLE_FRACTION)
    print("="*76)
    print("  SIZING BOUNDARY — flat N micros, fixed 2:1 baseline")
    print("  breach@ = trade# the $2000 trailing MLL is first hit (None=survives)")
    print("="*76)
    for scope,seq in [("FULL",per1),("OOS",per1[cut:])]:
        print(f"\n  --- {scope} ({len(seq)} trades) ---")
        print(f"  {'micros':>6}{'net$':>10}{'maxDD$':>10}{'PF':>6}{'win%':>6}{'breach@':>9}")
        for m in (1,2,3,4):
            d=diag([x*m for x in seq])
            b=f"#{d['breach']}" if d['breach'] else "survives"
            print(f"  {m:>6}{d['net']:>10.0f}{d['maxdd']:>10.0f}{d['pf']:>6.2f}{d['wr']*100:>6.0f}{b:>9}")

if __name__=="__main__":
    main()
