"""
Multi-instrument honest sweep. Runs a representative strategy grid on EACH instrument
(correct per-instrument $ economics), with train/val/holdout, and ONE multiple-testing-
corrected Reality-Check bar across the ENTIRE search (instruments x combos).

Self-contained (computes P&L in POINTS, converts with instruments.py economics) so it
does not depend on modular.py's hardcoded MES point value.

Honesty: testing 8 instruments x N combos = lots of chances for a luck winner. The
Reality-Check bar is the 95th pct of the best-of-(TOTAL combos tried across ALL
instruments) random strategy. A survivor must clear THAT. Accept the verdict.

Run: python -m backtest.multisweep
"""
import sys, random
from pathlib import Path
from dataclasses import dataclass
from itertools import product
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

from backtest.data import load_cached
from backtest.instruments import INSTRUMENTS, EVAL_CLEAN, EVAL_FLAGGED
from backtest.config import COMMISSION_RT, SLIPPAGE_TICKS, TRAILING_MLL

ATR_PERIOD=14; EMA_TREND=10
ENTRY_START=9*60+35; ENTRY_END=12*60; FLAT_MIN=15*60+55; RTH_START=9*60+30; RTH_END=16*60

OUT=open("backtest/multisweep_results.txt","w")
def log(m=""):
    print(m); OUT.write(m+"\n"); OUT.flush()

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

@dataclass(frozen=True)
class P:
    entry:str; exit:str; stop_atr:float; bN:int
    def label(self): return f"{self.entry}/{self.exit}/s{self.stop_atr}/N{self.bN}"

ENTRIES=["breakout","donchian","ema_pull"]
EXITS=["fixed_2r","fixed_1_5r","chandelier","time"]
STOPS=[1.5,2.5]
BREAKN=[6,12]

def make_grid():
    return [P(e,x,s,n) for e,x,s,n in product(ENTRIES,EXITS,STOPS,BREAKN)]

def to_sessions(df):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["date"]=df["timestamp"].dt.date.astype(str)
    df["mins"]=df["timestamp"].dt.hour*60+df["timestamp"].dt.minute
    rth=df[(df["mins"]>=RTH_START)&(df["mins"]<RTH_END)]
    out=[]
    for d,g in rth.groupby("date"):
        g=g.reset_index(drop=True)
        if len(g)>=ATR_PERIOD+5: out.append((d,g))
    return out

def run_combo_points(sessions, p):
    """Return list of per-trade POINTS (gross). One trade/day."""
    res=[]
    for d,g in sessions:
        o=g["open"].to_numpy(); h=g["high"].to_numpy(); l=g["low"].to_numpy(); c=g["close"].to_numpy()
        mins=g["mins"].to_numpy(); n=len(g)
        ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD)
        for i in range(ATR_PERIOD+1,n-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(atr[i]) or np.isnan(ema[i]): continue
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            sig=None
            if p.entry in ("breakout","donchian"):
                N=p.bN if p.entry=="breakout" else max(p.bN,20)
                if i<N: continue
                if up and c[i]>h[i-N:i].max(): sig="long"
                elif dn and c[i]<l[i-N:i].min(): sig="short"
            elif p.entry=="ema_pull":
                if up and l[i]<=ema[i] and c[i]>ema[i] and c[i]>o[i]: sig="long"
                elif dn and h[i]>=ema[i] and c[i]<ema[i] and c[i]<o[i]: sig="short"
            if sig is None: continue
            entry=o[i+1]; a=atr[i]; sp=p.stop_atr*a
            stop=entry-sp if sig=="long" else entry+sp
            rmult = 2.0 if p.exit=="fixed_2r" else (1.5 if p.exit=="fixed_1_5r" else None)
            tgt = (entry+rmult*sp if sig=="long" else entry-rmult*sp) if rmult else None
            hh=entry; exit_px=c[-1]
            for j in range(i+1,n):
                hh=max(hh,h[j]) if sig=="long" else min(hh,l[j])
                if p.exit=="chandelier":
                    stop = max(stop,hh-p.stop_atr*atr[j]) if sig=="long" else min(stop,hh+p.stop_atr*atr[j])
                if mins[j]>=FLAT_MIN: exit_px=c[j]; break
                if sig=="long":
                    if l[j]<=stop: exit_px=stop; break
                    if tgt and h[j]>=tgt: exit_px=tgt; break
                else:
                    if h[j]>=stop: exit_px=stop; break
                    if tgt and l[j]<=tgt: exit_px=tgt; break
            res.append((exit_px-entry) if sig=="long" else (entry-exit_px))
            break
    return res

def pts_to_net(points, econ):
    pv=econ["point_value"]; tick=econ["tick_size"]
    slip=SLIPPAGE_TICKS*tick*pv*2
    return [pt*pv - COMMISSION_RT - slip for pt in points]

def stats(nets):
    n=len(nets)
    if n==0: return None
    a=np.asarray(nets,float); wins=a[a>0]; gl=-a[a<=0].sum()
    pf=(wins.sum()/gl) if gl>0 else float("inf")
    from backtest.metrics import simulate_mll
    breach,_=simulate_mll(list(a))
    return {"n":n,"net":float(a.sum()),"pf":pf,"wr":len(wins)/n,"breach":breach,"exp":float(a.mean())}

def main():
    random.seed(11)
    grid=make_grid()
    instruments=["NQ","YM","RTY","GC","CL","SI","ZB","ZN"]
    K_total = len(grid)*len(instruments)   # total strategies tried across whole search
    log("="*80)
    log("  MULTI-INSTRUMENT HONEST SWEEP")
    log(f"  {len(instruments)} instruments x {len(grid)} combos = {K_total} total strategies")
    log("  Reality-Check bar scales with the WHOLE search (can't luck into a winner).")
    log("="*80)

    all_survivors=[]   # (inst, P, holdout stats)
    holdout_random_pools={}  # inst -> per-day random outcomes (for RC bar)

    for inst in instruments:
        econ=INSTRUMENTS[inst]
        clean = "clean" if inst in EVAL_CLEAN else "FLAGGED-econ"
        try:
            df=load_cached(Path("backtest/cache")/f"{inst}_5min_24h.parquet")
        except FileNotFoundError:
            log(f"\n{inst}: no cache, skip"); continue
        sess=to_sessions(df)
        dates=[d for d,_ in sess]; n=len(dates)
        i_tr=int(n*0.5); i_va=int(n*0.7)
        train=sess[:i_tr]; val=sess[i_tr:i_va]; hold=sess[i_va:]

        # rank on train
        rows=[]
        for p in grid:
            s=stats(pts_to_net(run_combo_points(train,p),econ))
            if s and s["n"]>=25: rows.append((p,s))
        rows.sort(key=lambda r:r[1]["pf"],reverse=True)
        # validation survivors
        cands=[]
        for p,tr in rows[:8]:
            if tr["pf"]<=1.1: continue
            va=stats(pts_to_net(run_combo_points(val,p),econ))
            if va and va["n"]>=10 and va["pf"]>1.1 and va["pf"]>=0.6*tr["pf"]:
                cands.append((p,tr,va))
        best_line = f"{inst:4s} [{clean:11s}] trainTop PF {rows[0][1]['pf']:.2f} ({rows[0][0].label()})" if rows else f"{inst}: no combos"
        log("\n"+best_line)
        log(f"        survived validation: {len(cands)}")
        # build holdout random pool for RC
        pool=[]
        for d,g in hold:
            o=g["open"].to_numpy(); h=g["high"].to_numpy(); l=g["low"].to_numpy(); c=g["close"].to_numpy()
            mins=g["mins"].to_numpy(); atr=_atr(h,l,c,ATR_PERIOD); outs=[]
            for i in range(ATR_PERIOD+1,len(g)-1):
                if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
                if np.isnan(atr[i]): continue
                sp=2.5*atr[i]
                for sd in ("long","short"):
                    entry=o[i+1]; stop=entry-sp if sd=="long" else entry+sp
                    tgt=entry+2*sp if sd=="long" else entry-2*sp; ex=c[-1]
                    for j in range(i+1,len(g)):
                        if mins[j]>=FLAT_MIN: ex=c[j]; break
                        if sd=="long":
                            if l[j]<=stop: ex=stop; break
                            if h[j]>=tgt: ex=tgt; break
                        else:
                            if h[j]>=stop: ex=stop; break
                            if l[j]<=tgt: ex=tgt; break
                    outs.append(((ex-entry) if sd=="long" else (entry-ex)))
            if outs: pool.append(pts_to_net(outs,econ))
        holdout_random_pools[inst]=pool
        for p,tr,va in cands:
            ho=stats(pts_to_net(run_combo_points(hold,p),econ))
            if ho: all_survivors.append((inst,p,tr,va,ho))

    # --- Reality Check across whole search ---
    log("\n"+"="*80)
    log("  HOLDOUT SURVIVORS + cross-search Reality Check")
    log("="*80)
    if not all_survivors:
        log("  NO instrument produced a validation survivor. VERDICT: NO-GO across all 8.")
        OUT.close(); return

    # RC bar: best-of-K_total random. Approx by, per sim, sampling one random strategy
    # net from a random instrument's pool and taking max over K_total draws.
    pools_flat=[pool for pool in holdout_random_pools.values() if pool]
    def one_random_net():
        pool=random.choice(pools_flat)
        return sum(random.choice(day) for day in pool)
    NSIM=1500
    bok=np.array([max(one_random_net() for _ in range(K_total)) for _ in range(NSIM)])
    bar=float(np.percentile(bok,95))
    log(f"  Reality-Check bar (95th pct of best-of-{K_total} random across all): ${bar:.0f}\n")

    any_go=False
    for inst,p,tr,va,ho in sorted(all_survivors,key=lambda x:x[4]["net"],reverse=True):
        flag = "" if inst in EVAL_CLEAN else " [ECON-FLAGGED]"
        passed = ho["net"]>bar and not ho["breach"] and ho["pf"]>1.3
        any_go=any_go or (passed and inst in EVAL_CLEAN)
        log(f"  {inst:4s}{flag} {p.label():<26} holdout: n={ho['n']} PF={ho['pf']:.2f} "
            f"net=${ho['net']:.0f} win={ho['wr']*100:.0f}% MLL={'BREACH' if ho['breach'] else 'ok'} "
            f"-> {'GO' if passed else 'no'}")
    log("\n"+"="*80)
    log(f"  VERDICT: {'GO on a clean instrument — see above' if any_go else 'NO-GO across all 8 instruments'}")
    log("  (GO requires: beat cross-search RC bar, PF>1.3, no MLL breach, clean econ.)")
    log("="*80)
    OUT.close()

if __name__=="__main__":
    main()
