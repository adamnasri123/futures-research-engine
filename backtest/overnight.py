"""
OVERNIGHT-LIQUIDITY strategies — now WITH real overnight pools (24h cache).

The thesis (user's, + ICT/"smart money" framing): the overnight Globex high/low are
major resting-stop pools. RTH price often SWEEPS one (runs the stops) then REVERSES —
the "judas swing." This is the test the prior liquidity run couldn't do (RTH-only).

Each RTH session, we know (causally, from data before the open):
  ONH / ONL = overnight high / low (18:00 prior day -> 09:30 today, ET)
  PDH / PDL = prior RTH-day high / low

Setups (long/short):
  on_sweep_rev : RTH bar pokes BELOW ONL (or above ONH) then closes back inside ->
                 fade it (enter reversal toward the opposite pool).
  on_break     : RTH closes beyond ONH/ONL and holds -> momentum continuation (control
                 for the opposite hypothesis).
  pd_sweep_rev : same but using prior-day H/L pools.

Exits: fixed 2R, opposite-pool target (liquidity draw), swing-trail, chandelier.

Honesty: 3-way split (train/val/holdout) + White's Reality Check bar scaled to the
grid size. Accept the verdict.

Run: python -m backtest.overnight
"""
import sys, random
from pathlib import Path
from dataclasses import dataclass
from itertools import product
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd

from backtest.data import load_cached_24h
from backtest.costs import apply_costs
from backtest.metrics import simulate_mll
from backtest.config import POINT_VALUE, SLIPPAGE_TICKS

ATR_PERIOD = 14
SWING_K = 2
RTH_START = 9*60+30
RTH_END   = 16*60
ENTRY_START = 9*60+35
ENTRY_END   = 12*60
FLAT_MIN    = 15*60+55
SWEEP_TOL_ATR = 0.10


def _atr(h,l,c,p):
    n=len(h); tr=np.empty(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=np.full(n,np.nan)
    if n>=p:
        a[p-1]=tr[:p].mean()
        for i in range(p,n): a[i]=(a[i-1]*(p-1)+tr[i])/p
    return a


def _swings(h,l,k):
    n=len(h); sh,sl=[],[]
    for i in range(k,n-k):
        if h[i]==h[i-k:i+k+1].max() and h[i]>h[i-1] and h[i]>h[i+1]: sh.append((i,h[i],i+k))
        if l[i]==l[i-k:i+k+1].min() and l[i]<l[i-1] and l[i]<l[i+1]: sl.append((i,l[i],i+k))
    return sh,sl


@dataclass(frozen=True)
class P:
    entry:str; exit:str; stop_atr:float
    def label(self): return f"{self.entry}->{self.exit} s{self.stop_atr}"


def build_sessions(df):
    """Split the 24h frame into RTH sessions, each tagged with overnight + prior-day
    pools computed ONLY from data available before that RTH open (causal)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["mins"] = df["timestamp"].dt.hour*60 + df["timestamp"].dt.minute
    df["is_rth"] = (df["mins"] >= RTH_START) & (df["mins"] < RTH_END)
    rth_dates = sorted(df[df["is_rth"]]["date"].unique())

    sessions = []
    prev_rth_hi = prev_rth_lo = None
    for d in rth_dates:
        rth = df[(df["date"]==d) & (df["is_rth"])].reset_index(drop=True)
        if len(rth) < ATR_PERIOD+5:
            # still update prior-day for the next session
            if len(rth):
                prev_rth_hi=float(rth["high"].max()); prev_rth_lo=float(rth["low"].min())
            continue
        # overnight = bars strictly BEFORE today's RTH open, back to prior 18:00 ET
        open_ts = rth["timestamp"].iloc[0]
        on_start = open_ts - pd.Timedelta(hours=15, minutes=30)  # ~18:00 prior day
        on = df[(df["timestamp"] >= on_start) & (df["timestamp"] < open_ts)]
        onh = float(on["high"].max()) if len(on) else None
        onl = float(on["low"].min()) if len(on) else None
        sessions.append({
            "date": d, "rth": rth, "onh": onh, "onl": onl,
            "pdh": prev_rth_hi, "pdl": prev_rth_lo,
        })
        prev_rth_hi=float(rth["high"].max()); prev_rth_lo=float(rth["low"].min())
    return sessions


def run_session(s, p:P):
    rth = s["rth"]
    o=rth["open"].to_numpy(); h=rth["high"].to_numpy()
    l=rth["low"].to_numpy(); c=rth["close"].to_numpy()
    mins=(rth["timestamp"].dt.hour*60+rth["timestamp"].dt.minute).to_numpy()
    n=len(rth)
    atr=_atr(h,l,c,ATR_PERIOD)
    sh,sl=_swings(h,l,SWING_K)

    if p.entry in ("on_sweep_rev","on_break"):
        hi_pool, lo_pool = s["onh"], s["onl"]
    else:  # pd_sweep_rev
        hi_pool, lo_pool = s["pdh"], s["pdl"]
    if hi_pool is None or lo_pool is None:
        return None

    for i in range(ATR_PERIOD+1, n-1):
        if mins[i] < ENTRY_START or mins[i] > ENTRY_END: continue
        if np.isnan(atr[i]): continue
        tol = SWEEP_TOL_ATR*atr[i]
        side=None
        if p.entry in ("on_sweep_rev","pd_sweep_rev"):
            # fade a failed poke through a pool
            if l[i] < lo_pool - tol and c[i] > lo_pool: side="long"
            elif h[i] > hi_pool + tol and c[i] < hi_pool: side="short"
        elif p.entry == "on_break":
            if c[i] > hi_pool: side="long"
            elif c[i] < lo_pool: side="short"
        if side is None: continue

        res=_manage(o,h,l,c,mins,atr,sh,sl,i+1,side,p,hi_pool,lo_pool)
        if res is None: return None
        return apply_costs(res[0], SLIPPAGE_TICKS)
    return None


def _manage(o,h,l,c,mins,atr,sh,sl,ei,side,p,hi_pool,lo_pool):
    n=len(c); entry=o[ei]
    a0=atr[ei-1] if not np.isnan(atr[ei-1]) else atr[ei]
    stop = entry-p.stop_atr*a0 if side=="long" else entry+p.stop_atr*a0
    risk=abs(entry-stop)
    if risk<=0: return None
    if p.exit=="fixed_2r":
        tgt = entry+2*risk if side=="long" else entry-2*risk
    elif p.exit=="liq_target":
        tgt = hi_pool if side=="long" else lo_pool   # draw toward opposite pool
    else:
        tgt=None
    hh=entry
    for i in range(ei,n):
        hh=max(hh,h[i]) if side=="long" else min(hh,l[i])
        if p.exit=="swing_trail":
            if side=="long":
                sw=max([px for (idx,px,conf) in sl if conf<=i and px<c[i]],default=None)
                if sw: stop=max(stop,sw)
            else:
                sw=min([px for (idx,px,conf) in sh if conf<=i and px>c[i]],default=None)
                if sw: stop=min(stop,sw)
        elif p.exit=="chandelier":
            stop = max(stop,hh-p.stop_atr*atr[i]) if side=="long" else min(stop,hh+p.stop_atr*atr[i])
        if side=="long" and l[i]<=stop: return (stop-entry)*POINT_VALUE,"stop"
        if side=="short" and h[i]>=stop: return (entry-stop)*POINT_VALUE,"stop"
        if tgt is not None:
            if side=="long" and h[i]>=tgt: return (tgt-entry)*POINT_VALUE,"target"
            if side=="short" and l[i]<=tgt: return (entry-tgt)*POINT_VALUE,"target"
        if mins[i]>=FLAT_MIN:
            px=c[i]; return ((px-entry) if side=="long" else (entry-px))*POINT_VALUE,"time"
    px=c[-1]; return ((px-entry) if side=="long" else (entry-px))*POINT_VALUE,"end"


def run_combo(sessions,p,dates):
    nets=[]
    for s in sessions:
        if s["date"] not in dates: continue
        r=run_session(s,p)
        if r is not None: nets.append(r)
    return nets


def stats(nets):
    n=len(nets)
    if n==0: return None
    a=np.asarray(nets,float); wins=a[a>0]; gl=-a[a<=0].sum()
    pf=(wins.sum()/gl) if gl>0 else float("inf")
    breach,_=simulate_mll(list(a))
    return {"n":n,"net":float(a.sum()),"exp":float(a.mean()),"pf":pf,"wr":len(wins)/n,"breach":breach}


def main():
    random.seed(1)
    print("="*72)
    print("  OVERNIGHT-LIQUIDITY SWEEP  (now WITH real overnight pools)")
    print("="*72)
    df=load_cached_24h()
    sessions=build_sessions(df)
    dates=[s["date"] for s in sessions]
    n=len(dates)
    i_tr=int(n*0.5); i_va=int(n*0.7)
    train,val,hold=set(dates[:i_tr]),set(dates[i_tr:i_va]),set(dates[i_va:])
    print(f"\n{n} RTH sessions w/ overnight pools | train {len(train)} val {len(val)} holdout {len(hold)}")

    entries=["on_sweep_rev","pd_sweep_rev","on_break"]
    exits=["fixed_2r","liq_target","swing_trail","chandelier"]
    stops=[1.0,1.5,2.0,2.5,3.0]
    grid=[P(e,x,s) for e,x,s in product(entries,exits,stops)]
    K=len(grid)
    print(f"Grid: {K} strategies")

    rows=[]
    for p in grid:
        st=stats(run_combo(sessions,p,train))
        if st and st["n"]>=25: rows.append((p,st))
    rows.sort(key=lambda r:r[1]["pf"],reverse=True)
    print(f"\nTop 10 on TRAIN (of {len(rows)} with >=25 trades):")
    print(f"  {'strategy':<30}{'n':>4}{'PF':>6}{'net$':>8}{'win%':>6}")
    for p,st in rows[:10]:
        print(f"  {p.label():<30}{st['n']:>4}{st['pf']:>6.2f}{st['net']:>8.0f}{st['wr']*100:>6.0f}")

    cands=[]
    for p,tr in rows[:15]:
        if tr["pf"]<=1.1: continue
        va=stats(run_combo(sessions,p,val))
        if va and va["n"]>=10 and va["pf"]>1.1 and va["pf"]>=0.6*tr["pf"]:
            cands.append((p,tr,va))
    print(f"\nSurvived validation: {len(cands)}")
    for p,tr,va in cands[:10]:
        print(f"  {p.label():<30} trPF {tr['pf']:.2f}  vaPF {va['pf']:.2f}")
    if not cands:
        print("\nNO candidate generalized train->val. VERDICT: NO-GO.")
        return

    # Reality Check on holdout
    proxy=P("on_break","fixed_2r",2.0)
    per_day=[]
    for s in sessions:
        if s["date"] not in hold: continue
        rth=s["rth"]; o=rth["open"].to_numpy(); h=rth["high"].to_numpy()
        l=rth["low"].to_numpy(); c=rth["close"].to_numpy()
        mins=(rth["timestamp"].dt.hour*60+rth["timestamp"].dt.minute).to_numpy()
        atr=_atr(h,l,c,ATR_PERIOD); sh,sl=_swings(h,l,SWING_K)
        hp,lp=s["onh"],s["onl"]
        if hp is None or lp is None: continue
        outs=[]
        for i in range(ATR_PERIOD+1,len(rth)-1):
            if mins[i]<ENTRY_START or mins[i]>ENTRY_END: continue
            if np.isnan(atr[i]): continue
            for sd in ("long","short"):
                r=_manage(o,h,l,c,mins,atr,sh,sl,i+1,sd,proxy,hp,lp)
                if r is not None: outs.append(apply_costs(r[0],SLIPPAGE_TICKS))
        if outs: per_day.append(outs)

    def rand_net(): return sum(random.choice(o) for o in per_day)
    NSIM=2000
    bok=np.array([max(rand_net() for _ in range(K)) for _ in range(NSIM)])
    bar=float(np.percentile(bok,95))

    best=cands[0][0]
    ho=stats(run_combo(sessions,best,hold))
    print(f"\n--- HOLDOUT + Reality Check (K={K}) ---")
    print(f"  best: {best.label()}")
    print(f"  holdout: n={ho['n']} PF={ho['pf']:.2f} net=${ho['net']:.0f} win={ho['wr']*100:.0f}% MLLbreach={'YES' if ho['breach'] else 'no'}")
    print(f"  Reality-Check bar (95th pct best-of-{K} random): ${bar:.0f}")
    passed = ho["net"]>bar and not ho["breach"] and ho["pf"]>1.3
    print(f"\n  VERDICT: {'GO — beat the corrected bar' if passed else 'NO-GO'}")


if __name__=="__main__":
    main()
