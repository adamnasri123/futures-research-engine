"""
High-impact US economic-event calendar for the backtest window (Jun 2024 - May 2026).

Forex Factory's free JSON feed is current-week only, so it cannot drive a 2-year
backtest. Instead we use deterministic / published sources:

  - FOMC announcement days: published by the Federal Reserve (fixed schedule).
    Hardcoded below; announcement is at 2:00 PM ET (DURING our session).
  - NFP (jobs report): first Friday of each month, 8:30 AM ET (before the open).

NOTE: these FOMC dates are entered from the Fed's published calendar to the best
of available knowledge. CPI is intentionally omitted (its date drifts and a wrong
date would just add noise). Treat news results as exploratory given the small
event count, not as a validated edge.
"""
from datetime import date, timedelta

# FOMC statement/announcement dates (2nd day of each meeting), ET.
FOMC_DAYS = {
    # 2024
    "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled — full year, verified against federalreserve.gov 2026-06-09)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}

FOMC_ANNOUNCE_MIN = 14 * 60  # 2:00 PM ET


def _first_friday(year: int, month: int) -> str:
    d = date(year, month, 1)
    # weekday(): Mon=0 .. Fri=4
    offset = (4 - d.weekday()) % 7
    return (d + timedelta(days=offset)).isoformat()


def nfp_days_for_range(start_iso: str, end_iso: str) -> set:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    out = set()
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        ff = _first_friday(y, m)
        if start_iso <= ff <= end_iso:
            out.add(ff)
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def tag_days(dates: list) -> dict:
    """Return {date: set_of_tags} with tags in {'fomc','nfp'}."""
    if not dates:
        return {}
    nfp = nfp_days_for_range(min(dates), max(dates))
    out = {}
    for d in dates:
        tags = set()
        if d in FOMC_DAYS:
            tags.add("fomc")
        if d in nfp:
            tags.add("nfp")
        out[d] = tags
    return out
