import requests

TRADES_URL = "https://api.topstepx.com/api/Trade/search"


def search_trades(token: str, account_id: int, start_timestamp: str, end_timestamp: str = None) -> list[dict]:
    """Search filled trades for an account in a time window.
    Each trade has: profitAndLoss (None = half-turn), fees, side, size, price, ..."""
    response = requests.post(
        TRADES_URL,
        headers={
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"accountId": account_id, "startTimestamp": start_timestamp, "endTimestamp": end_timestamp},
    )
    response.raise_for_status()
    data = response.json()
    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"Trade search failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )
    return data["trades"]
