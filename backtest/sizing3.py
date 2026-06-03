"""
User's REFINED idea, tested properly (combining what was tested separately before):

  EXIT = trail stop BELOW nearest support (confirmed swing) AND take profit AT the
         nearest LIQUIDITY pool (overnight H/L + prior-day H/L), but ONLY take the
         trade if that pool is NOT too far (reward <= MAX_TGT_ATR * ATR). "Don't chase
         liquidity that's way up there."  -> structure-based exit, not an arbitrary 2:1.

  Plus: flat-1 vs $200 volatility sizing.

Uses the 24h cache so liquidity pools are REAL (incl. overnight). Honest validation:
full + OOS, $2000 trailing-MLL breach is decisive. PLUS breach DIAGNOSTICS: worst
losing streak and how fast the drawdown actually accumulates (to settle the "it'd take
10 days, we'd adapt" question with data).

Run: python -m backtest.sizing3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.costs import apply_costs
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS, IN_SAMPLE_FRACTION, TRAILING_MLL

EMA_TREND=10; BREAKOUT_N=6; ATR_PERIOD=14; STOP_ATR=2.5
RTH_START=9*60+30; RTH_END=16*60
ENTRY_START=9*60+35; ENTRY_END=12*60; FLAT_MIN=15*60+55
RISK_TARGET_USD=200.0; MAX_CONTRACTS=10
MAX_TGT_ATR=6.0     # "reward not too high up": skip if nearest pool > this*ATR away

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


def build_sessions(df):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["date"]=df["timestamp"].dt.date.astype(str)
    df["mins"]=df["timestamp"].dt.hour*60+df["timestamp"].dt.minute
    df["is_rth"]=(df["mins"]>=RTH_START)&(df["mins"]<RTH_END)
    rth_dates=sorted(df[df["is_rth"]]["date"].unique())
    sessions=[]; phi=plo=None
    for d in rth_dates:
        rth=df[(df["date"]==d)&(df["is_rth"])].reset_index(drop=True)
        if len(rth)<ATR_PERIOD+5:
            if len(rth): phi=float(rth["high"].max()); plo=float(rth["low"].min())
            continue
        open_ts=rth["timestamp"].iloc[0]
        on=df[(df["timestamp"]>=open_ts-pd.Timedelta(hours=15,minutes=30))&(df["timestamp"]<open_ts)]
        onh=float(on["high"].max()) if len(on) else None
        onl=float(on["low"].min()) if len(on) else None
        sessions.append({"date":d,"rth":rth,"onh":onh,"onl":onl,"pdh":phi,"pdl":plo})
        phi=float(rth["high"].max()); plo=float(rth["low"].min())
    return sessions


def replay_combined(sessions):
    """Trail under support + target at nearest liquidity pool, skip if pool too far."""
    trades=[]
    for s in sessions:
        rth=s["rth"]
        o=rth["open"].to_numpy(); h=rth["high"].to_numpy()
        l=rth["low"].to_numpy(); c=rth["close"].to_numpy()
        mins=(rth["timestamp"].dt.hour*60+rth["timestamp"].dt.minute).to_numpy()
        n=len(rth); ema=_ema(c,EMA_TREND); atr=_atr(h,l,c,ATR_PERIOD); sh,sl=_swings(h,l)
        pools_hi=[x for x in [s["onh"],s["pdh"]] if x]
        pools_lo=[x for x in [s["onl"],s["pdl"]] if x]
        for i in range(ATR_PERIOD+1,n-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(ema[i]) or np.isnan(atr[i]): continue
            up=c[i]>ema[i] and ema[i]>ema[i-1]; dn=c[i]<ema[i] and ema[i]<ema[i-1]
            sig=None
            if up and c[i]>h[i-BREAKOUT_N:i].max(): sig="long"
            elif dn and c[i]<l[i-BREAKOUT_N:i].min(): sig="short"
            if sig is None: continue
            entry=o[i+1]; a=atr[i]
            # target = nearest pool in trade direction; skip if too far or none
            if sig=="long":
                above=[p for p in pools_hi if p>entry]
                tgt=min(above) if above else None
            else:
                below=[p for p in pools_lo if p<entry]
                tgt=max(below) if below else None
            if tgt is None or abs(tgt-entry) > MAX_TGT_ATR*a:
                continue   # reward too far / no pool -> stand aside (user's rule)
            # initial stop below/above structure: nearest swing, else ATR
            stop = entry-STOP_ATR*a if sig=="long" else entry+STOP_ATR*a
            stop_pts=abs(entry-stop)
            trail=stop; exit_px=c[-1]
            for j in range(i+1,n):
                if mins[j]>=FLAT_MIN: exit_px=c[j]; break
                if sig=="long":
                    sw=max([px for (idx,px,cf) in sl if cf<=j and px<c[j]],default=None)
                    if sw: trail=max(trail,sw)
                    if l[j]<=trail: exit_px=trail; break
                    if h[j]>=tgt: exit_px=tgt; break
                else:
                    sw=min([px for (idx,px,cf) in sh if cf<=j and px>c[j]],default=None)
                    if sw: trail=min(trail,sw)
                    if h[j]>=trail: exit_px=trail; break
                    if l[j]<=tgt: exit_px=tgt; break
            pts=(exit_px-entry) if sig=="long" else (entry-exit_px)
            trades.append((pts, stop_pts))
            break
    return trades


def mll_diag(nets):
    """Return (breached, trade_idx_of_breach, worst_losing_streak)."""
    eq=0.0; peak=0.0; mll=-TRAILING_MLL
    streak=0; worst_streak=0; breach_at=None
    for k,p in enumerate(nets):
        eq+=p
        streak = streak+1 if p<=0 else 0
        worst_streak=max(worst_streak,streak)
        if eq>peak: peak=eq; mll=peak-TRAILING_MLL
        if breach_at is None and eq<mll: breach_at=k+1
    return breach_at, worst_streak


def evaluate(trades, sizing, oos):
    if oos:
        cut=int(len(trades)*IN_SAMPLE_FRACTION); trades=trades[cut:]
    nets=[]; sizes=[]
    for pts,stop_pts in trades:
        size=1 if sizing=="flat" else int(max(1,min(MAX_CONTRACTS,round(RISK_TARGET_USD/(stop_pts*POINT_VALUE))))) if stop_pts>0 else 1
        nets.append(apply_costs(pts*POINT_VALUE,SLIPPAGE_TICKS)*size); sizes.append(size)
    a=np.asarray(nets,float)
    if not len(a): return None
    wins=a[a>0]; gl=-a[a<=0].sum(); pf=(wins.sum()/gl) if gl>0 else float("inf")
    eq=np.cumsum(a); dd=float((eq-np.maximum.accumulate(eq)).min())
    breach_at, worst=mll_diag(list(a))
    return {"n":len(a),"net":float(a.sum()),"exp":float(a.mean()),"pf":pf,"wr":len(wins)/len(a),
            "maxdd":dd,"breach_at":breach_at,"worst_streak":worst,"avgsize":float(np.mean(sizes))}


def main():
    df=load_cached_24h(); sessions=build_sessions(df)
    trades=replay_combined(sessions)
    print("="*82)
    print("  COMBINED EXIT: trail-under-support + target-at-liquidity (real overnight pools)")
    print(f"  {len(trades)} trades taken (skips when nearest pool > {MAX_TGT_ATR}xATR away)")
    print("  EVAL REALITY: $2000 trailing-MLL breach = auto-fail.")
    print("="*82)
    print(f"  {'sizing':<8}{'scope':<6}{'n':>4}{'avgSz':>6}{'net$':>9}{'exp$':>7}{'PF':>6}{'win%':>6}{'maxDD$':>9}{'breach@':>9}{'wStreak':>8}")
    for sizing in ["flat","vol200"]:
        for scope,oos in [("full",False),("OOS",True)]:
            s=evaluate(trades,sizing,oos)
            if s:
                b = f"#{s['breach_at']}" if s['breach_at'] else "no"
                print(f"  {sizing:<8}{scope:<6}{s['n']:>4}{s['avgsize']:>6.1f}{s['net']:>9.0f}"
                      f"{s['exp']:>7.1f}{s['pf']:>6.2f}{s['wr']*100:>6.0f}{s['maxdd']:>9.0f}{b:>9}{s['worst_streak']:>8}")
    print("\n  breach@ = trade number where the $2000 trailing limit was first breached")
    print("            (1 trade/day, so trade# ~ trading day# — settles 'would take 10 days')")
    print("  wStreak = worst consecutive-losing-trade run (the cluster that causes breaches)")

if __name__=="__main__":
    main()
