"""
Fetch 1-MINUTE ES bars (full ~24h session) from Massive.com into a separate cache.
Needed for the 1-min structure/SAR strategy tests (2026-06-10 user idea).
Same contract-stitching approach as data.py; ~1380 bars/day -> ~700k rows total.
Rate limit 5 calls/min; pagination 50k bars/page -> ~2 pages per quarter.

Run: python -m backtest.data_1min
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.config import CACHE_DIR

CACHE_FILE = Path(CACHE_DIR) / "ES_1min_24h.parquet"
BASE_URL = "https://api.massive.com"
HEADERS = {"Authorization": f"Bearer {os.getenv('MASSIVE_API_KEY')}"}
SLEEP_BETWEEN = 13

# Same roll windows as data.py, with ESM6 extended through today.
CONTRACTS = [
    ("ESM4", "2024-06-10", "2024-06-21"),
    ("ESU4", "2024-06-21", "2024-09-20"),
    ("ESZ4", "2024-09-20", "2024-12-20"),
    ("ESH5", "2024-12-20", "2025-03-21"),
    ("ESM5", "2025-03-21", "2025-06-20"),
    ("ESU5", "2025-06-20", "2025-09-19"),
    ("ESZ5", "2025-09-19", "2025-12-19"),
    ("ESH6", "2025-12-19", "2026-03-21"),
    ("ESM6", "2026-03-21", "2026-06-10"),
]


def _fetch(ticker, start, end):
    bars = []
    params = {"resolution": "1min", "window_start.gte": start,
              "window_start.lte": end, "limit": 50000}
    url = f"{BASE_URL}/futures/v1/aggs/{ticker}"
    while url:
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 429:
            print("  rate limited - 60s..."); time.sleep(60); continue
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            print(f"  warning: {ticker} status={data.get('status')}"); break
        bars.extend(data.get("results", []))
        url = data.get("next_url"); params = {}
        if url:
            time.sleep(SLEEP_BETWEEN)   # pagination calls also count against the limit
    return bars


def main():
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (tk, s, e) in enumerate(CONTRACTS):
        print(f"[{i+1}/{len(CONTRACTS)}] {tk} {s}..{e} ", end="", flush=True)
        bars = _fetch(tk, s, e)
        print(f"{len(bars)} bars")
        for b in bars:
            rows.append({
                "timestamp": pd.Timestamp(b["window_start"], unit="ns", tz="UTC")
                               .tz_convert("America/New_York"),
                "open": b["open"], "high": b["high"], "low": b["low"],
                "close": b["close"], "volume": b["volume"], "ticker": tk,
            })
        if i < len(CONTRACTS) - 1:
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data fetched.")
    df = df[df["timestamp"].dt.weekday < 5]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(CACHE_FILE, index=False)
    days = df["timestamp"].dt.date.nunique()
    print(f"\nCached {len(df)} 1-min bars / {days} days -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
