"""
Phase A: Fetch 5-minute ES futures bars from Massive.com.
Stitches quarterly contracts into a continuous series, filters to RTH (9:30-11:30 ET),
caches to parquet. Rate limit: 5 calls/min → 13s sleep between requests.
One call per quarter = ~8 calls total = ~2 minutes.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.config import CACHE_DIR

CACHE_PATH = Path(CACHE_DIR)
CACHE_FILE  = CACHE_PATH / "ES_5min.parquet"
CACHE_FILE_24H = CACHE_PATH / "ES_5min_24h.parquet"  # full session incl. overnight

BASE_URL = "https://api.massive.com"
API_KEY  = os.getenv("MASSIVE_API_KEY")
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

SLEEP_BETWEEN = 13   # seconds — stays under 5 calls/min

# Quarterly contract periods (ticker, period_start, period_end)
# Using front-month roll dates (approximate third Friday of expiry month)
CONTRACTS = [
    ("ESM4", "2024-06-10", "2024-06-21"),   # partial — start of our window
    ("ESU4", "2024-06-21", "2024-09-20"),
    ("ESZ4", "2024-09-20", "2024-12-20"),
    ("ESH5", "2024-12-20", "2025-03-21"),
    ("ESM5", "2025-03-21", "2025-06-20"),
    ("ESU5", "2025-06-20", "2025-09-19"),
    ("ESZ5", "2025-09-19", "2025-12-19"),
    ("ESH6", "2025-12-19", "2026-03-21"),
    ("ESM6", "2026-03-21", "2026-05-31"),
]

# RTH filter: keep the full cash session 9:30–16:00 ET (minute-aware).
# Strategies locate the 9:30 bar by timestamp — never assume bar[0] is 9:30.
RTH_START_MIN = 9 * 60 + 30   # 9:30 ET
RTH_END_MIN   = 16 * 60       # 16:00 ET (exclusive)


def _fetch_quarter(ticker: str, start: str, end: str) -> list[dict]:
    """Fetch all 5-min bars for one quarterly contract, handling pagination."""
    bars = []
    params = {
        "resolution":          "5min",
        "window_start.gte":    start,
        "window_start.lte":    end,
        "limit":               50000,
    }
    url = f"{BASE_URL}/futures/v1/aggs/{ticker}"

    while url:
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 429:
            print("  Rate limited — waiting 60s...")
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, params=params)

        r.raise_for_status()
        data = r.json()

        if data.get("status") != "OK":
            print(f"  Warning: {ticker} returned status={data.get('status')}")
            break

        batch = data.get("results", [])
        bars.extend(batch)

        # Pagination
        url    = data.get("next_url")
        params = {}   # next_url already includes params

    return bars


def _ns_to_et(ns: int) -> pd.Timestamp:
    """Convert nanosecond Unix timestamp to US/Eastern timezone."""
    return pd.Timestamp(ns, unit="ns", tz="UTC").tz_convert("America/New_York")


def fetch_and_cache() -> pd.DataFrame:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    all_rows = []
    total = len(CONTRACTS)

    for i, (ticker, start, end) in enumerate(CONTRACTS):
        print(f"  [{i+1}/{total}] {ticker}  {start} to {end} ...", end=" ", flush=True)
        bars = _fetch_quarter(ticker, start, end)
        print(f"{len(bars)} bars raw")

        for b in bars:
            ts = _ns_to_et(b["window_start"])
            all_rows.append({
                "timestamp": ts,
                "open":      b["open"],
                "high":      b["high"],
                "low":       b["low"],
                "close":     b["close"],
                "volume":    b["volume"],
                "ticker":    ticker,
            })

        if i < total - 1:
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No data fetched. Check API key and ticker formats.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to RTH 9:30–16:00 ET on weekdays only (minute-aware)
    df = df[df["timestamp"].dt.weekday < 5]
    mins = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(mins >= RTH_START_MIN) & (mins < RTH_END_MIN)]

    # Drop duplicates at roll dates (both contracts may cover same day)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(CACHE_FILE, index=False)
    print(f"\nCached {len(df)} RTH bars to {CACHE_FILE}")
    return df


def fetch_and_cache_24h() -> pd.DataFrame:
    """Same fetch as fetch_and_cache, but keeps the FULL ~24h session (incl. overnight
    Globex) so strategies can use overnight-high/low liquidity pools. Separate cache
    file — does NOT touch the proven RTH cache."""
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    all_rows = []
    total = len(CONTRACTS)
    for i, (ticker, start, end) in enumerate(CONTRACTS):
        print(f"  [{i+1}/{total}] {ticker}  {start} to {end} ...", end=" ", flush=True)
        bars = _fetch_quarter(ticker, start, end)
        print(f"{len(bars)} bars raw")
        for b in bars:
            all_rows.append({
                "timestamp": _ns_to_et(b["window_start"]),
                "open": b["open"], "high": b["high"], "low": b["low"],
                "close": b["close"], "volume": b["volume"], "ticker": ticker,
            })
        if i < total - 1:
            time.sleep(SLEEP_BETWEEN)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No data fetched.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.weekday < 5]                 # weekdays only
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(CACHE_FILE_24H, index=False)
    print(f"\nCached {len(df)} full-session bars to {CACHE_FILE_24H}")
    return df


def load_cached(path=CACHE_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"No cache at {path}. Run fetch first.")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df


def load_cached_24h() -> pd.DataFrame:
    return load_cached(CACHE_FILE_24H)


def get_trading_days(df: pd.DataFrame) -> list[str]:
    return sorted(df["timestamp"].dt.date.astype(str).unique().tolist())


def group_by_day(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """
    Pre-group bars into per-day DataFrames ONCE.
    Returns a list of (date_str, day_bars) sorted by date.
    This avoids re-filtering the full frame on every iteration (the perf killer).
    """
    df = df.sort_values("timestamp")
    day_key = df["timestamp"].dt.date.astype(str)
    groups = []
    for date, day_bars in df.groupby(day_key, sort=True):
        groups.append((date, day_bars.reset_index(drop=True)))
    return groups


if __name__ == "__main__":
    print("Fetching 5-min ES bars from Massive.com (~2 years, 9 quarters)...")
    print("Rate limit: 5 calls/min — expect ~2 minutes total.\n")
    df = fetch_and_cache()
    days = get_trading_days(df)
    print(f"\nData summary:")
    print(f"  Date range   : {days[0]} to {days[-1]}")
    print(f"  Trading days : {len(days)}")
    print(f"  Total bars   : {len(df)}")
    print(f"  Bars/day avg : {len(df)/len(days):.1f}")
    print(f"\nSample (first 6 bars):")
    print(df.head(6).to_string())
