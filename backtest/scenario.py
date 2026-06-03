"""
SCENARIO-ADAPTIVE strategy (user's idea): default = breakout+2:1, BUT when price is
at a major higher-timeframe support/resistance level, switch to a "bounce" play
(long off support / short off resistance) targeting the move back toward range.

HTF levels are built from PRIOR days only (causal — no hindsight). "Major support" =
a prior swing low / prior-day low / overnight low that price is now testing. The
example ("4h chart hit support then ran up") is exactly this — but in real time we
only know the level from history, not that it will hold.

Compares: baseline (breakout 2:1 always) vs scenario-adaptive (bounce at HTF levels,
breakout otherwise). Honest OOS + MLL. If the bounce sub-strategy has a SHALLOWER
drawdown, it could justify larger size — that's the real prize, noted explicitly.

Run: python -m backtest.scenario
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION, TRAILING_MLL

EMA_TREND=10; BREAKOUT_N=6; ATR_PERIOD=14; STOP_ATR=2.5
RTH_START=9*60+30; RTH_END=16*60
ENTRY_START=9*60+35; ENTRY_END=12*60; FLAT_MIN=15*60+55
NEAR_ATR=0.75          # "at" a level = within this*ATR of it
LOOKBACK_DAYS=10       # HTF levels from the last N prior days

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


def build(df):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["date"]=df["timestamp"].dt.date.astype(str)
    df["mins"]=df["timestamp"].dt.hour*60+df["timestamp"].dt.minute
    df["is_rth"]=(df["mins"]>=RTH_START)&(df["mins"]<RTH_END)
    rdates=sorted(df[df["is_rth"]]["date"].unique())
    daily={}
    for d in rdates:
        r=df[(df["date"]==d)&df["is_rth"]]
        daily[d]=(float(r["high"].max()),float(r["low"].min()))
    sessions=[]
    for ix,d in enumerate(rdates):
        r=df[(df["date"]==d)&df["is_rth"]].reset_index(drop=True)
        if len(r)<ATR_PERIOD+5: continue
        # HTF levels = highs/lows of the prior LOOKBACK_DAYS days (causal)
        prior=rdates[max(0,ix-LOOKBACK_DAYS):ix]
        levels=[]
        for pd_ in prior:
            hi,lo=daily[pd_]; levels.extend([hi,lo])
        sessions.append({"date":d,"rth":r,"levels":sorted(set(levels))})
    return sessions


def run_session(s, mode):
    """mode='baseline' (breakout 2:1) or 'scenario' (bounce at HTF level else breakout)."""
    r=s["rth"]; o=r["open"].to_numpy(); h=r["high"].to_numpy()
    l=r["low"].to_numpy(); c=r["close"].to_numpy()
    mins=(r["timestamp"].dt.hour*60+r["timestamp"].dt.minute).to_numpy()
    n=len(r); ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD)
    levels=s["levels"]
    for i in range(ATR_PERIOD+1,n-1):
        if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
        if np.isnan(ema[i]) or np.isnan(atr[i]): continue
        a=atr[i]
        # nearest HTF level to current price
        near=None
        for lv in levels:
            if abs(c[i]-lv) <= NEAR_ATR*a:
                near=lv; break
        sig=None; mode_used=None
        if mode=="scenario" and near is not None:
            # BOUNCE play: long if testing a level from above-ish & closing up, etc.
            if c[i] > near and l[i] <= near + NEAR_ATR*a and c[i] > o[i]:
                sig="long"; mode_used="bounce"
            elif c[i] < near and h[i] >= near - NEAR_ATR*a and c[i] < o[i]:
                sig="short"; mode_used="bounce"
        if sig is None:
            # breakout default
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            if up and c[i]>h[i-BREAKOUT_N:i].max(): sig="long"; mode_used="breakout"
            elif dn and c[i]<l[i-BREAKOUT_N:i].min(): sig="short"; mode_used="breakout"
        if sig is None: continue
        entry=o[i+1]; stop_pts=STOP_ATR*a
        stop = entry-stop_pts if sig=="long" else entry+stop_pts
        tgt  = entry+2*stop_pts if sig=="long" else entry-2*stop_pts
        exit_px=c[-1]
        for j in range(i+1,n):
            if mins[j]>=FLAT_MIN: exit_px=c[j]; break
            if sig=="long":
                if l[j]<=stop: exit_px=stop; break
                if h[j]>=tgt: exit_px=tgt; break
            else:
                if h[j]>=stop: exit_px=stop; break
                if l[j]<=tgt: exit_px=tgt; break
        pts=(exit_px-entry) if sig=="long" else (entry-exit_px)
        return apply_costs(pts*POINT_VALUE,SLIPPAGE_TICKS), mode_used
    return None


def diag(nets):
    a=np.asarray(nets,float)
    if not len(a): return None
    wins=a[a>0]; gl=-a[a<=0].sum(); pf=(wins.sum()/gl) if gl>0 else float("inf")
    dd=float((np.cumsum(a)-np.maximum.accumulate(np.cumsum(a))).min())
    breach,_=simulate_mll(list(a))
    return {"n":len(a),"net":float(a.sum()),"exp":float(a.mean()),"pf":pf,
            "wr":len(wins)/len(a),"maxdd":dd,"breach":breach}


def main():
    df=load_cached_24h(); sessions=build(df)
    dates=[s["date"] for s in sessions]; cut=int(len(dates)*IN_SAMPLE_FRACTION)
    print("="*80)
    print("  SCENARIO-ADAPTIVE (bounce at HTF support/resistance) vs BASELINE breakout")
    print("  HTF levels = highs/lows of prior 10 days (causal, no hindsight)")
    print("="*80)
    for mode in ["baseline","scenario"]:
        res=[]; modes={"bounce":0,"breakout":0}
        for s in sessions:
            r=run_session(s,mode)
            if r is not None:
                res.append(r[0]); modes[r[1]]=modes.get(r[1],0)+1
        full=diag(res); oos=diag(res[cut:])
        print(f"\n  MODE = {mode}   (bounce trades: {modes.get('bounce',0)}, breakout trades: {modes.get('breakout',0)})")
        for label,s in [("full",full),("OOS",oos)]:
            if s:
                print(f"    {label:<5} n={s['n']:>3} net=${s['net']:>7.0f} exp=${s['exp']:>6.1f} "
                      f"PF={s['pf']:.2f} win={s['wr']*100:>3.0f}% maxDD=${s['maxdd']:>7.0f} "
                      f"breach={'YES' if s['breach'] else 'no'}")
    print("\n  KEY: does 'scenario' beat 'baseline' OOS AND have shallower maxDD?")
    print("  (shallower maxDD is what could justify >1 micro — the real prize.)")

if __name__=="__main__":
    main()
