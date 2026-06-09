"""
Does entering EARLIER help? Tests the user's "it enters late" intuition honestly.

Current live = "breakout_confirmed": wait for a 5-min bar to CLOSE beyond the last
6-bar high/low, enter NEXT bar open. ~5-10 min lag.

Variants tested (same stop/target/day-filter, same costs, OOS):
  A. confirmed (live baseline) — close-beyond, enter next open
  B. intrabar         — enter the INSTANT price pierces the 6-bar level (no wait for
                        close). Earliest possible; risks fakeouts.
  C. enter_on_break_close — enter at the CLOSE of the breakout bar itself (skip the
                        next-bar-open wait). ~5 min earlier than baseline.

All on the 507-day RTH cache, OOS = last 30%, vs random-entry benchmark. We judge:
does earlier entry beat baseline OOS, and does it still beat random?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import random
from backtest.data import load_cached, group_by_day
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION

EMA_TREND=10; BREAKOUT_N=6; ATR_PERIOD=14; STOP_ATR=2.5
ENTRY_START=9*60+35; ENTRY_END=12*60; FLAT_MIN=15*60+55

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

def run(groups, mode):
    """mode: 'confirmed' | 'intrabar' | 'break_close'. One trade/day, fixed 2:1 stop."""
    nets=[]
    for date, day in groups:
        o=day["open"].to_numpy(); h=day["high"].to_numpy()
        l=day["low"].to_numpy(); c=day["close"].to_numpy()
        mins=(day["timestamp"].dt.hour*60+day["timestamp"].dt.minute).to_numpy()
        n=len(day)
        if n<ATR_PERIOD+3: continue
        ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD)
        for i in range(ATR_PERIOD+1,n-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(ema[i]) or np.isnan(atr[i]): continue
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            hi6=h[i-BREAKOUT_N:i].max(); lo6=l[i-BREAKOUT_N:i].min()
            sig=None; entry=None
            if mode=="confirmed":
                # need CLOSE beyond level; enter next bar open
                if up and c[i]>hi6: sig="long"; entry=o[i+1]; ei=i+1
                elif dn and c[i]<lo6: sig="short"; entry=o[i+1]; ei=i+1
            elif mode=="break_close":
                # enter at the close of the breakout bar (5 min earlier, same bar)
                if up and c[i]>hi6: sig="long"; entry=c[i]; ei=i+1
                elif dn and c[i]<lo6: sig="short"; entry=c[i]; ei=i+1
            elif mode=="intrabar":
                # enter the instant the level is pierced (bias from prior bar to stay causal)
                up_p=c[i-1]>ema[i-1] and ema[i-1]>ema[i-2] if i>=2 else False
                dn_p=c[i-1]<ema[i-1] and ema[i-1]<ema[i-2] if i>=2 else False
                if up_p and h[i]>hi6: sig="long"; entry=hi6; ei=i+1   # filled at the level
                elif dn_p and l[i]<lo6: sig="short"; entry=lo6; ei=i+1
            if sig is None: continue
            a=atr[i]; sp=STOP_ATR*a
            stop = entry-sp if sig=="long" else entry+sp
            tgt = entry+2*sp if sig=="long" else entry-2*sp
            exit_px=c[-1]
            for j in range(ei,n):
                if mins[j]>=FLAT_MIN: exit_px=c[j]; break
                if sig=="long":
                    if l[j]<=stop: exit_px=stop; break
                    if h[j]>=tgt: exit_px=tgt; break
                else:
                    if h[j]>=stop: exit_px=stop; break
                    if l[j]<=tgt: exit_px=tgt; break
            pts=(exit_px-entry) if sig=="long" else (entry-exit_px)
            nets.append(apply_costs(pts*POINT_VALUE, SLIPPAGE_TICKS))
            break
    return nets

def stats(nets, oos=False):
    if oos:
        cut=int(len(nets)*IN_SAMPLE_FRACTION); nets=nets[cut:]
    a=np.asarray(nets,float)
    if not len(a): return None
    wins=a[a>0]; gl=-a[a<=0].sum(); pf=(wins.sum()/gl) if gl>0 else float("inf")
    dd=float((np.cumsum(a)-np.maximum.accumulate(np.cumsum(a))).min())
    breach,_=simulate_mll(list(a))
    return {"n":len(a),"net":float(a.sum()),"exp":float(a.mean()),"pf":pf,
            "wr":len(wins)/len(a),"maxdd":dd,"breach":breach}

def main():
    groups=group_by_day(load_cached())
    print("="*74)
    print("  ENTRY TIMING TEST — does entering EARLIER help? (1 micro, fixed 2:1)")
    print("="*74)
    print(f"  {'mode':<14}{'scope':<6}{'n':>4}{'net$':>9}{'exp$':>7}{'PF':>6}{'win%':>6}{'maxDD$':>9}{'breach':>8}")
    results={}
    for mode in ["confirmed","break_close","intrabar"]:
        nets=run(groups,mode); results[mode]=nets
        for scope,oos in [("full",False),("OOS",True)]:
            s=stats(nets,oos)
            if s:
                tag = " <-- LIVE" if (mode=="confirmed" and scope=="OOS") else ""
                print(f"  {mode:<14}{scope:<6}{s['n']:>4}{s['net']:>9.0f}{s['exp']:>7.1f}"
                      f"{s['pf']:>6.2f}{s['wr']*100:>6.0f}{s['maxdd']:>9.0f}{('YES' if s['breach'] else 'no'):>8}{tag}")
    print("\n  Read: does break_close/intrabar BEAT confirmed on OOS net AND PF?")
    print("  If not clearly better, 'late entry' isn't the problem — it's just a no-edge coin.")

if __name__=="__main__":
    main()
