"""
Multi-instrument data fetcher (resumable). Stitches quarterly/monthly contracts per
instrument into a continuous 5-min series, caches one parquet per instrument.

CRITICAL: wrong contract months = silently corrupt data. So after stitching we VALIDATE
(bar counts per quarter + price jump at each roll boundary) and flag anything suspicious
instead of trusting it. Each instrument cached separately so a rate-limit hiccup doesn't
lose prior progress.

Run:  python -m backtest.multidata          # fetch all, skip already-cached
      python -m backtest.multidata GC NQ    # fetch only these
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.config import CACHE_DIR

CACHE_PATH = Path(CACHE_DIR)
BASE_URL = "https://api.massive.com"
HEADERS = {"Authorization": f"Bearer {os.getenv('MASSIVE_API_KEY')}"}
SLEEP = 13

# Standard window (matches ES cache span)
QSTART = "2024-06-10"

# Quarterly roll windows (H=Mar,M=Jun,U=Sep,Z=Dec) — used by equity indices + bonds.
Q_WINDOWS = [
    ("H4", "2024-06-10", "2024-06-21"),  # placeholder start uses M4 below; see per-instrument
]

# Per-instrument contract series: (ticker, start, end). Year is single-digit.
# Equity indices & bonds share ES's quarterly H/M/U/Z structure and roll dates.
_QUARTERLY_ROLLS = [
    ("M4", "2024-06-10", "2024-06-21"),
    ("U4", "2024-06-21", "2024-09-20"),
    ("Z4", "2024-09-20", "2024-12-20"),
    ("H5", "2024-12-20", "2025-03-21"),
    ("M5", "2025-03-21", "2025-06-20"),
    ("U5", "2025-06-20", "2025-09-19"),
    ("Z5", "2025-09-19", "2025-12-19"),
    ("H6", "2025-12-19", "2026-03-21"),
    ("M6", "2026-03-21", "2026-05-31"),
]

def _quarterly(root):
    return [(f"{root}{code}", s, e) for code, s, e in _QUARTERLY_ROLLS]

# Gold (GC) liquid cycle ~ G,J,M,Q,V,Z. Use Q(Aug)/Z(Dec)/J(Apr)/M(Jun)... approximate
# bi-monthly fronts spanning the window.
_GC_ROLLS = [
    ("Q4", "2024-06-10", "2024-07-28"),
    ("Z4", "2024-07-28", "2024-11-25"),
    ("J5", "2024-11-25", "2025-03-27"),
    ("M5", "2025-03-27", "2025-05-27"),
    ("Q5", "2025-05-27", "2025-07-28"),
    ("Z5", "2025-07-28", "2025-11-24"),
    ("J6", "2025-11-24", "2026-03-27"),
    ("M6", "2026-03-27", "2026-05-31"),
]
# Crude (CL) is monthly — front month rolls ~20th. Approximate with monthly contracts.
_CL_MONTHS = [
    ("N4","2024-06-10","2024-06-20"),("Q4","2024-06-20","2024-07-22"),("U4","2024-07-22","2024-08-20"),
    ("V4","2024-08-20","2024-09-20"),("X4","2024-09-20","2024-10-21"),("Z4","2024-10-21","2024-11-20"),
    ("F5","2024-11-20","2024-12-19"),("G5","2024-12-19","2025-01-21"),("H5","2025-01-21","2025-02-20"),
    ("J5","2025-02-20","2025-03-20"),("K5","2025-03-20","2025-04-22"),("M5","2025-04-22","2025-05-20"),
    ("N5","2025-05-20","2025-06-20"),("Q5","2025-06-20","2025-07-22"),("U5","2025-07-22","2025-08-20"),
    ("V5","2025-08-20","2025-09-22"),("X5","2025-09-22","2025-10-21"),("Z5","2025-10-21","2025-11-20"),
    ("F6","2025-11-20","2025-12-19"),("G6","2025-12-19","2026-01-21"),("H6","2026-01-21","2026-02-20"),
    ("J6","2026-02-20","2026-03-20"),("K6","2026-03-20","2026-04-21"),("M6","2026-04-21","2026-05-20"),
    ("N6","2026-05-20","2026-05-31"),
]
# Silver (SI) ~ H,K,N,U,Z cycle
_SI_ROLLS = [
    ("N4","2024-06-10","2024-07-26"),("U4","2024-07-26","2024-09-25"),("Z4","2024-09-25","2024-12-26"),
    ("H5","2024-12-26","2025-03-26"),("K5","2025-03-26","2025-05-27"),("N5","2025-05-27","2025-07-28"),
    ("U5","2025-07-28","2025-09-25"),("Z5","2025-09-25","2025-12-26"),("H6","2025-12-26","2026-03-26"),
    ("K6","2026-03-26","2026-05-27"),("N6","2026-05-27","2026-05-31"),
]

INSTRUMENTS = {
    "NQ":  _quarterly("NQ"),
    "YM":  _quarterly("YM"),
    "RTY": _quarterly("RTY"),
    "ZB":  _quarterly("ZB"),
    "ZN":  _quarterly("ZN"),
    "GC":  [(f"GC{c}", s, e) for c, s, e in _GC_ROLLS],
    "CL":  [(f"CL{c}", s, e) for c, s, e in _CL_MONTHS],
    "SI":  [(f"SI{c}", s, e) for c, s, e in _SI_ROLLS],
}


def _fetch(ticker, start, end):
    bars=[]
    params={"resolution":"5min","window_start.gte":start,"window_start.lte":end,"limit":50000}
    url=f"{BASE_URL}/futures/v1/aggs/{ticker}"
    while url:
        for attempt in range(4):
            r=requests.get(url,headers=HEADERS,params=params)
            if r.status_code==429:
                time.sleep(20); continue
            break
        d=r.json()
        if d.get("status")!="OK":
            return bars, d.get("status")
        bars.extend(d.get("results",[]))
        url=d.get("next_url"); params={}
    return bars, "OK"


def fetch_instrument(name):
    series=INSTRUMENTS[name]
    rows=[]; per_q=[]
    for tk,s,e in series:
        b,status=_fetch(tk,s,e)
        per_q.append((tk,len(b),status))
        for x in b:
            rows.append({"timestamp":pd.Timestamp(x["window_start"],unit="ns",tz="UTC").tz_convert("America/New_York"),
                         "open":x["open"],"high":x["high"],"low":x["low"],"close":x["close"],
                         "volume":x["volume"],"ticker":tk})
        time.sleep(SLEEP)
    if not rows:
        return None, per_q
    df=pd.DataFrame(rows)
    df=df[df["timestamp"].dt.weekday<5]
    df=df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out=CACHE_PATH/f"{name}_5min_24h.parquet"
    df.to_parquet(out,index=False)
    return df, per_q


def validate(name, df, per_q):
    """Flag suspicious stitches: empty quarters, or big price jumps at roll boundaries."""
    issues=[]
    for tk,nb,st in per_q:
        if st!="OK": issues.append(f"{tk}: status={st}")
        elif nb==0: issues.append(f"{tk}: 0 bars")
    # roll-boundary jumps
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["d"]=df["ticker"].ne(df["ticker"].shift())
    boundaries=df.index[df["d"]].tolist()[1:]
    for bi in boundaries:
        if bi<1: continue
        prev=df["close"].iloc[bi-1]; cur=df["open"].iloc[bi]
        if prev>0 and abs(cur-prev)/prev > 0.05:   # >5% gap at roll = suspicious
            issues.append(f"roll @ {str(df['timestamp'].iloc[bi])[:10]} {df['ticker'].iloc[bi-1]}->{df['ticker'].iloc[bi]}: {prev:.1f}->{cur:.1f} ({(cur-prev)/prev*100:+.1f}%)")
    return issues


def main():
    CACHE_PATH.mkdir(parents=True,exist_ok=True)
    want = sys.argv[1:] if len(sys.argv)>1 else list(INSTRUMENTS.keys())
    print(f"Fetching: {want}")
    summary=[]
    for name in want:
        out=CACHE_PATH/f"{name}_5min_24h.parquet"
        if out.exists():
            print(f"  {name}: already cached, skipping (delete to refetch)")
            continue
        print(f"  {name}: fetching {len(INSTRUMENTS[name])} contracts...", flush=True)
        df,per_q=fetch_instrument(name)
        if df is None:
            print(f"    FAILED — no data. per-contract: {per_q}")
            summary.append((name,0,["no data"])); continue
        days=df["timestamp"].dt.date.nunique()
        issues=validate(name,df,per_q)
        print(f"    cached {len(df)} bars, {days} days. issues: {len(issues)}")
        for iss in issues[:6]: print(f"      ! {iss}")
        summary.append((name,days,issues))
    print("\n=== SUMMARY ===")
    for name,days,issues in summary:
        flag = "OK" if not issues else f"{len(issues)} ISSUES"
        print(f"  {name:5s} {days:4d} days  [{flag}]")


if __name__=="__main__":
    main()
