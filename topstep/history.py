import requests
from datetime import datetime, timezone

BASE_URL = "https://api.topstepx.com/api/History/retrieveBars"

# Bar units
SECOND = 1
MINUTE = 2
HOUR   = 3
DAY    = 4
WEEK   = 5
MONTH  = 6


def _headers(token: str) -> dict:
    return {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def get_bars(
    token: str,
    contract_id: str,
    start_time: str,
    end_time: str,
    unit: int = MINUTE,
    unit_number: int = 1,
    limit: int = 500,
    live: bool = False,
    include_partial: bool = False,
) -> list[dict]:
    """
    Fetch OHLCV bars. Each bar has keys: t (time), o, h, l, c, v.
    Max 20,000 bars per request. Rate limit: 50 requests / 30 seconds.
    """
    r = requests.post(
        BASE_URL,
        headers=_headers(token),
        json={
            "contractId": contract_id,
            "live": live,
            "startTime": start_time,
            "endTime": end_time,
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": include_partial,
        },
    )
    r.raise_for_status()
    data = r.json()

    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"Retrieve bars failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )

    return data["bars"]


if __name__ == "__main__":
    from auth import get_session_token
    from contracts import CONTRACTS

    token = get_session_token()
    bars = get_bars(
        token,
        contract_id=CONTRACTS["MES"],
        start_time="2026-05-01T00:00:00Z",
        end_time="2026-05-30T00:00:00Z",
        unit=DAY,
        unit_number=1,
        limit=30,
    )
    print(f"Got {len(bars)} bars:\n")
    for b in bars:
        print(f"  {b['t']}  O={b['o']}  H={b['h']}  L={b['l']}  C={b['c']}  V={b['v']}")
