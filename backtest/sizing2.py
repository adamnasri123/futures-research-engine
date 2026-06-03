"""
User's proposal, tested honestly:
  (A) Volatility sizing — risk ~$200/trade (more micros when stop is tight), vs flat 1.
  (B) "Let it ride" exits — instead of fixed 2:1, ride until structure break / nearest
      liquidity pool, only capping when reward is reasonable.

KEY EVAL REALITY: the $2000 trailing drawdown AUTO-FAILS the account. So the headline
P&L is meaningless if a scheme breaches the MLL — that = eval over. We judge every
scheme on: net, AND max drawdown, AND (decisive) does it breach $2000 trailing.

Sizing is applied to the SAME trade sequence (sizing can't change which trades happen),
but the EXIT schemes DO change per-contract outcomes, so each exit is replayed fresh.

Run: python -m backtest.sizing2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from backtest.data import load_cached, group_by_day
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, TICK_SIZE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION

EMA_TREND=10; BREAKOUT_N=6; ATR_PERIOD=14; STOP_ATR=2.5
ENTRY_START=9*60+35; ENTRY_END=12*60; FLAT_MIN=15*60+55
RISK_TARGET_USD=200.0          # user's requested per-trade risk
MAX_CONTRACTS=10               # hard cap (eval safety)

def _ema(a,p):
    o=np.full(len(a),np.nan)
    if not len(a): return o
    k=2/(p+1); o[0]=a[0]
    for i in range(1,len(a)): o[i]=a[i]*k+o[i-1]*(1-k)
    return o
def _atr(h,l,c,p):
    n=len(h); tr=np.empty(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=np.full(n,np.nan)
    if n>=p:
        a[p-1]=tr[:p].mean()
        for i in range(p,n): a[i]=(a[i-1]*(p-1)+tr[i])/p
    return a
def _swings(h,l,k=2):
    n=len(h); sh,sl=[],[]
    for i in range(k,n-k):
        if h[i]==h[i-k:i+k+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh.append((i,h[i],i+k))
        if l[i]==l[i-k:i+k+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl.append((i,l[i],i+k))
    return sh,sl


def replay(groups, exit_mode):
    """Return list of trades: each = (pts_per_contract_gross, stop_pts). exit_mode in
    {'fixed_2r','ride_struct','ride_liq'}."""
    trades=[]
    prev_hi=prev_lo=None
    for date, day in groups:
        o=day["open"].to_numpy(); h=day["high"].to_numpy()
        l=day["low"].to_numpy(); c=day["close"].to_numpy()
        mins=(day["timestamp"].dt.hour*60+day["timestamp"].dt.minute).to_numpy()
        n=len(day)
        pdh,pdl=prev_hi,prev_lo
        prev_hi=float(h.max()); prev_lo=float(l.min())
        if n<ATR_PERIOD+5: continue
        ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD); sh,sl=_swings(h,l)
        for i in range(ATR_PERIOD+1,n-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(ema[i]) or np.isnan(atr[i]): continue
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            sig=None
            if up and c[i]>h[i-BREAKOUT_N:i].max(): sig="long"
            elif dn and c[i]<l[i-BREAKOUT_N:i].min(): sig="short"
            if sig is None: continue
            entry=o[i+1]; a=atr[i]; stop_pts=STOP_ATR*a
            stop_px = entry-stop_pts if sig=="long" else entry+stop_pts
            # fixed target only for fixed_2r
            tgt_px = (entry+2*stop_pts if sig=="long" else entry-2*stop_pts) if exit_mode=="fixed_2r" else None
            # liquidity target = nearest prior-day pool in trade direction
            if exit_mode=="ride_liq":
                if sig=="long": tgt_px = pdh if (pdh and pdh>entry) else None
                else:           tgt_px = pdl if (pdl and pdl<entry) else None
            trail=stop_px
            exit_px=c[-1]
            for j in range(i+1,n):
                if mins[j]>=FLAT_MIN: exit_px=c[j]; break
                # ride_struct: trail under last confirmed swing
                if exit_mode=="ride_struct":
                    if sig=="long":
                        sw=max([px for (idx,px,cf) in sl if cf<=j and px<c[j]],default=None)
                        if sw: trail=max(trail,sw)
                    else:
                        sw=min([px for (idx,px,cf) in sh if cf<=j and px>c[j]],default=None)
                        if sw: trail=min(trail,sw)
                stop_now = trail if exit_mode=="ride_struct" else stop_px
                if sig=="long":
                    if l[j]<=stop_now: exit_px=stop_now; break
                    if tgt_px and h[j]>=tgt_px: exit_px=tgt_px; break
                else:
                    if h[j]>=stop_now: exit_px=stop_now; break
                    if tgt_px and l[j]<=tgt_px: exit_px=tgt_px; break
            pts=(exit_px-entry) if sig=="long" else (entry-exit_px)
            trades.append((pts, stop_pts))
            break
    return trades


def evaluate(trades, sizing, oos_only=False):
    """sizing in {'flat','vol200'}. Returns stats incl. MLL breach."""
    if oos_only:
        cut=int(len(trades)*IN_SAMPLE_FRACTION); trades=trades[cut:]
    nets=[]; sizes=[]
    for pts, stop_pts in trades:
        if sizing=="flat":
            size=1
        else:  # vol200: contracts so that stop_pts*$5*size ~ $200
            risk1=stop_pts*POINT_VALUE
            size=int(max(1,min(MAX_CONTRACTS, round(RISK_TARGET_USD/risk1)))) if risk1>0 else 1
        per1=apply_costs(pts*POINT_VALUE, SLIPPAGE_TICKS)
        nets.append(per1*size); sizes.append(size)
    a=np.asarray(nets,float)
    if not len(a): return None
    wins=a[a>0]; gl=-a[a<=0].sum(); pf=(wins.sum()/gl) if gl>0 else float("inf")
    eq=np.cumsum(a); dd=float((eq-np.maximum.accumulate(eq)).min())
    breach,_=simulate_mll(list(a))
    return {"n":len(a),"net":float(a.sum()),"exp":float(a.mean()),"pf":pf,
            "wr":len(wins)/len(a),"maxdd":dd,"breach":breach,"avgsize":float(np.mean(sizes))}


def main():
    groups=group_by_day(load_cached())
    print("="*78)
    print("  USER PROPOSAL TEST: $200 vol-sizing + ride-to-structure/liquidity exits")
    print("  EVAL REALITY: a $2000 trailing-MLL breach = account auto-failed.")
    print("="*78)
    for exit_mode in ["fixed_2r","ride_struct","ride_liq"]:
        trades=replay(groups, exit_mode)
        print(f"\n### EXIT = {exit_mode}  ({len(trades)} trades) ###")
        print(f"  {'sizing':<8}{'scope':<6}{'n':>4}{'avgSz':>6}{'net$':>9}{'exp$':>7}{'PF':>6}{'win%':>6}{'maxDD$':>9}{'MLLbreach':>11}")
        for sizing in ["flat","vol200"]:
            for scope,oos in [("full",False),("OOS",True)]:
                s=evaluate(trades,sizing,oos_only=oos)
                if s:
                    print(f"  {sizing:<8}{scope:<6}{s['n']:>4}{s['avgsize']:>6.1f}{s['net']:>9.0f}"
                          f"{s['exp']:>7.1f}{s['pf']:>6.2f}{s['wr']*100:>6.0f}{s['maxdd']:>9.0f}"
                          f"{('YES' if s['breach'] else 'no'):>11}")
    print("\n"+"="*78)
    print("  Decision rule: a scheme is viable ONLY if MLLbreach=no on OOS.")
    print("  More size / wider rides that breach = eval failure, regardless of net.")
    print("="*78)

if __name__=="__main__":
    main()
