import requests

BASE_URL = "https://api.topstepx.com/api/Position"

POSITION_TYPE = {0: "Undefined", 1: "Long", 2: "Short"}


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


def search_open_positions(token: str, account_id: int) -> list[dict]:
    r = requests.post(
        f"{BASE_URL}/searchOpen",
        headers=_headers(token),
        json={"accountId": account_id},
    )
    r.raise_for_status()
    data = r.json()
    _check(data, "Search positions")
    return data["positions"]


def close_position(token: str, account_id: int, contract_id: str) -> None:
    r = requests.post(
        f"{BASE_URL}/closeContract",
        headers=_headers(token),
        json={"accountId": account_id, "contractId": contract_id},
    )
    r.raise_for_status()
    _check(r.json(), "Close position")


def partial_close_position(token: str, account_id: int, contract_id: str, size: int) -> None:
    r = requests.post(
        f"{BASE_URL}/partialCloseContract",
        headers=_headers(token),
        json={"accountId": account_id, "contractId": contract_id, "size": size},
    )
    r.raise_for_status()
    _check(r.json(), "Partial close position")
