"""
Full honest validation of the SCENARIO-ADAPTIVE strategy.

Sweeps its parameters (NEAR_ATR, LOOKBACK_DAYS, target_R) on TRAIN only, keeps those
that generalize to VAL, then tests the survivor ONCE on HOLDOUT against a
multiple-testing-corrected random benchmark (White's Reality Check, bar scaled to the
number of param combos tried). This is the test that catches "the knobs were fitted."

Accept the verdict, including NO-GO.

Run: python -m backtest.scenario_validate
"""
import sys, random
from pathlib import Path
from itertools import product
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS

EMA_TREND=10; BREAKOUT_N=6; ATR_PERIOD=14; STOP_ATR=2.5
RTH_START=9*60+30; RTH_END=16*60
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


def build(df, max_lookback):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["date"]=df["timestamp"].dt.date.astype(str)
    df["mins"]=df["timestamp"].dt.hour*60+df["timestamp"].dt.minute
    df["is_rth"]=(df["mins"]>=RTH_START)&(df["mins"]<RTH_END)
    rdates=sorted(df[df["is_rth"]]["date"].unique())
    daily={}
    for d in rdates:
        r=df[(df["date"]==d)&df["is_rth"]]
        daily[d]=(float(r["high"].max()),float(r["low"].min()))
    rows=[]
    for ix,d in enumerate(rdates):
        r=df[(df["date"]==d)&df["is_rth"]].reset_index(drop=True)
        if len(r)<ATR_PERIOD+5: continue
        rows.append({"date":d,"rth":r,"ix":ix})
    return rows, rdates, daily


def levels_for(rdates, daily, ix, lookback):
    prior=rdates[max(0,ix-lookback):ix]
    lv=[]
    for pd_ in prior:
        hi,lo=daily[pd_]; lv.extend([hi,lo])
    return sorted(set(lv))


def run_session(sess, rdates, daily, near_atr, lookback, tgt_r):
    r=sess["rth"]; o=r["open"].to_numpy(); h=r["high"].to_numpy()
    l=r["low"].to_numpy(); c=r["close"].to_numpy()
    mins=(r["timestamp"].dt.hour*60+r["timestamp"].dt.minute).to_numpy()
    n=len(r); ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD)
    levels=levels_for(rdates,daily,sess["ix"],lookback)
    for i in range(ATR_PERIOD+1,n-1):
        if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
        if np.isnan(ema[i]) or np.isnan(atr[i]): continue
        a=atr[i]; near=None
        for lvl in levels:
            if abs(c[i]-lvl)<=near_atr*a: near=lvl; break
        sig=None
        if near is not None:
            if c[i]>near and l[i]<=near+near_atr*a and c[i]>o[i]: sig="long"
            elif c[i]<near and h[i]>=near-near_atr*a and c[i]<o[i]: sig="short"
        if sig is None:
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            if up and c[i]>h[i-BREAKOUT_N:i].max(): sig="long"
            elif dn and c[i]<l[i-BREAKOUT_N:i].min(): sig="short"
        if sig is None: continue
        entry=o[i+1]; sp=STOP_ATR*a
        stop=entry-sp if sig=="long" else entry+sp
        tgt=entry+tgt_r*sp if sig=="long" else entry-tgt_r*sp
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
        return apply_costs(pts*POINT_VALUE,SLIPPAGE_TICKS)
    return None


def run_combo(rows, rdates, daily, params, dset):
    na,lb,tr=params
    nets=[]
    for s in rows:
        if s["date"] not in dset: continue
        r=run_session(s,rdates,daily,na,lb,tr)
        if r is not None: nets.append(r)
    return nets


def stats(nets):
    n=len(nets)
    if n==0: return None
    a=np.asarray(nets,float); wins=a[a>0]; gl=-a[a<=0].sum()
    pf=(wins.sum()/gl) if gl>0 else float("inf")
    breach,_=simulate_mll(list(a))
    return {"n":n,"net":float(a.sum()),"pf":pf,"wr":len(wins)/n,"breach":breach,"exp":float(a.mean())}


def main():
    random.seed(7)
    print("="*78)
    print("  SCENARIO-ADAPTIVE — FULL VALIDATION + Reality Check")
    print("="*78)
    df=load_cached_24h()
    MAXLB=20
    rows, rdates, daily = build(df, MAXLB)
    dates=[s["date"] for s in rows]; n=len(dates)
    i_tr=int(n*0.5); i_va=int(n*0.7)
    train,val,hold=set(dates[:i_tr]),set(dates[i_tr:i_va]),set(dates[i_va:])
    print(f"{n} sessions | train {len(train)} val {len(val)} holdout {len(hold)}")

    near_grid=[0.5,0.75,1.0,1.25]
    lb_grid=[5,10,15,20]
    tgt_grid=[1.5,2.0,2.5,3.0]
    grid=list(product(near_grid,lb_grid,tgt_grid))
    K=len(grid)
    print(f"Param grid: {K} combos (near {near_grid} x lookback {lb_grid} x target_R {tgt_grid})")

    rankings=[]
    for p in grid:
        s=stats(run_combo(rows,rdates,daily,p,train))
        if s and s["n"]>=30: rankings.append((p,s))
    rankings.sort(key=lambda r:r[1]["pf"],reverse=True)
    print(f"\nTop 8 on TRAIN (of {len(rankings)}):")
    print(f"  {'near/lb/tgt':<18}{'n':>4}{'PF':>6}{'net$':>8}{'win%':>6}")
    for p,s in rankings[:8]:
        print(f"  {str(p):<18}{s['n']:>4}{s['pf']:>6.2f}{s['net']:>8.0f}{s['wr']*100:>6.0f}")

    cands=[]
    for p,tr in rankings[:12]:
        if tr["pf"]<=1.1: continue
        va=stats(run_combo(rows,rdates,daily,p,val))
        if va and va["n"]>=12 and va["pf"]>1.1 and va["pf"]>=0.6*tr["pf"]:
            cands.append((p,tr,va))
    print(f"\nSurvived validation: {len(cands)}")
    for p,tr,va in cands[:8]:
        print(f"  {str(p):<18} trPF {tr['pf']:.2f} vaPF {va['pf']:.2f}")
    if not cands:
        print("\nNO param combo generalized train->val. VERDICT: NO-GO (knobs were fitted).")
        return

    # Reality Check on holdout
    best=cands[0][0]
    ho=stats(run_combo(rows,rdates,daily,best,hold))
    # random benchmark: random long/short at a random eligible bar, 2:1 exit, on holdout
    per_day=[]
    for s in rows:
        if s["date"] not in hold: continue
        r=s["rth"]; o=r["open"].to_numpy(); h=r["high"].to_numpy()
        l=r["low"].to_numpy(); c=r["close"].to_numpy()
        mins=(r["timestamp"].dt.hour*60+r["timestamp"].dt.minute).to_numpy()
        atr=_atr(h,l,c,ATR_PERIOD); outs=[]
        for i in range(ATR_PERIOD+1,len(r)-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(atr[i]): continue
            sp=STOP_ATR*atr[i]
            for sd in ("long","short"):
                entry=o[i+1]
                stop=entry-sp if sd=="long" else entry+sp
                tgt=entry+2*sp if sd=="long" else entry-2*sp
                ex=c[-1]
                for j in range(i+1,len(r)):
                    if mins[j]>=FLAT_MIN: ex=c[j]; break
                    if sd=="long":
                        if l[j]<=stop: ex=stop; break
                        if h[j]>=tgt: ex=tgt; break
                    else:
                        if h[j]>=stop: ex=stop; break
                        if l[j]<=tgt: ex=tgt; break
                pts=(ex-entry) if sd=="long" else (entry-ex)
                outs.append(apply_costs(pts*POINT_VALUE,SLIPPAGE_TICKS))
        if outs: per_day.append(outs)
    def rnet(): return sum(random.choice(o) for o in per_day)
    NSIM=2000
    bok=np.array([max(rnet() for _ in range(K)) for _ in range(NSIM)])
    bar=float(np.percentile(bok,95))

    print(f"\n--- HOLDOUT + Reality Check (K={K} param combos) ---")
    print(f"  best params: {best}")
    print(f"  holdout: n={ho['n']} PF={ho['pf']:.2f} net=${ho['net']:.0f} win={ho['wr']*100:.0f}% MLLbreach={'YES' if ho['breach'] else 'no'}")
    print(f"  Reality-Check bar (95th pct best-of-{K} random): ${bar:.0f}")
    passed = ho["net"]>bar and not ho["breach"] and ho["pf"]>1.3
    print(f"\n  VERDICT: {'GO — survived full validation + Reality Check!' if passed else 'NO-GO'}")
    if not passed:
        why=[]
        if ho["net"]<=bar: why.append(f"net ${ho['net']:.0f} <= bar ${bar:.0f}")
        if ho["breach"]: why.append("MLL breach")
        if ho["pf"]<=1.3: why.append(f"PF {ho['pf']:.2f} <= 1.3")
        print("  Reasons: " + "; ".join(why))
        print("  The OOS improvement seen earlier was the fitted knobs, not real edge.")

if __name__=="__main__":
    main()
