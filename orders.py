import requests

BASE_URL = "https://api.topstepx.com/api/Order"

# Order types
MARKET       = 2
LIMIT        = 1
STOP         = 4
TRAILING_STOP = 5
JOIN_BID     = 6
JOIN_ASK     = 7

# Order sides
BID = 0  # Buy
ASK = 1  # Sell

# Order statuses
STATUS = {0: "None", 1: "Open", 2: "Filled", 3: "Cancelled", 4: "Expired", 5: "Rejected", 6: "Pending"}


def _headers(token: str) -> dict:
    return {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _check(data: dict, label: str) -> None:
    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"{label} failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )


def place_order(
    token: str,
    account_id: int,
    contract_id: str,
    type: int,
    side: int,
    size: int,
    limit_price: float = None,
    stop_price: float = None,
    trail_price: float = None,
    custom_tag: str = None,
    stop_loss_ticks: int = None,
    take_profit_ticks: int = None,
) -> int:
    payload = {
        "accountId": account_id,
        "contractId": contract_id,
        "type": type,
        "side": side,
        "size": size,
        "limitPrice": limit_price,
        "stopPrice": stop_price,
        "trailPrice": trail_price,
        "customTag": custom_tag,
    }
    # Bracket ticks are SIGNED offsets from entry. For a long (BID), the stop sits
    # BELOW entry (negative) and target ABOVE (positive); for a short (ASK), mirror.
    # Callers pass positive magnitudes; we apply the correct sign here.
    if stop_loss_ticks is not None:
        mag = abs(stop_loss_ticks)
        sl = -mag if side == BID else mag
        payload["stopLossBracket"] = {"ticks": sl, "type": STOP}
    if take_profit_ticks is not None:
        mag = abs(take_profit_ticks)
        tp = mag if side == BID else -mag
        payload["takeProfitBracket"] = {"ticks": tp, "type": LIMIT}

    r = requests.post(f"{BASE_URL}/place", headers=_headers(token), json=payload)
    r.raise_for_status()
    data = r.json()
    _check(data, "Place order")
    return data["orderId"]


def cancel_order(token: str, account_id: int, order_id: int) -> None:
    r = requests.post(
        f"{BASE_URL}/cancel",
        headers=_headers(token),
        json={"accountId": account_id, "orderId": order_id},
    )
    r.raise_for_status()
    _check(r.json(), "Cancel order")


def modify_order(
    token: str,
    account_id: int,
    order_id: int,
    size: int = None,
    limit_price: float = None,
    stop_price: float = None,
    trail_price: float = None,
) -> None:
    r = requests.post(
        f"{BASE_URL}/modify",
        headers=_headers(token),
        json={
            "accountId": account_id,
            "orderId": order_id,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
        },
    )
    r.raise_for_status()
    _check(r.json(), "Modify order")


def search_orders(token: str, account_id: int, start_timestamp: str, end_timestamp: str = None) -> list[dict]:
    r = requests.post(
        f"{BASE_URL}/search",
        headers=_headers(token),
        json={"accountId": account_id, "startTimestamp": start_timestamp, "endTimestamp": end_timestamp},
    )
    r.raise_for_status()
    data = r.json()
    _check(data, "Search orders")
    return data["orders"]


def search_open_orders(token: str, account_id: int) -> list[dict]:
    r = requests.post(
        f"{BASE_URL}/searchOpen",
        headers=_headers(token),
        json={"accountId": account_id},
    )
    r.raise_for_status()
    data = r.json()
    _check(data, "Search open orders")
    return data["orders"]
