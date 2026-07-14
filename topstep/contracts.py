import requests
from topstep.auth import get_session_token

CONTRACTS_URL = "https://api.topstepx.com/api/Contract/available"

# Friendly name -> contract ID. STATIC FALLBACK ONLY — prefer resolve_contract(),
# which asks the API for the active (front-month) contract so quarterly rolls can
# never leave the bot polling an expired contract again (that happened 2026-07:
# the M26 hardcode expired in June and the bot sat on stale data).
CONTRACTS = {
    # Mini stock futures
    "ES":  "CON.F.US.EP.U26",   # S&P 500 Mini
    "NQ":  "CON.F.US.ENQ.U26",  # Nasdaq Mini
    "YM":  "CON.F.US.YM.U26",   # Dow Mini
    # Micro stock futures
    "MES": "CON.F.US.MES.U26",  # Micro S&P 500
    "MNQ": "CON.F.US.MNQ.U26",  # Micro Nasdaq
    "MYM": "CON.F.US.MYM.U26",  # Micro Dow
    # Metals
    "GC":  "CON.F.US.GCE.Q26",  # Gold
    "MGC": "CON.F.US.MGC.Q26",  # Micro Gold
    "SI":  "CON.F.US.SIE.N26",  # Silver
    "SIL": "CON.F.US.SIL.N26",  # Micro Silver
}

# Friendly name -> exchange symbol id (for dynamic front-month lookup)
SYMBOL_IDS = {
    "ES": "F.US.EP", "NQ": "F.US.ENQ", "YM": "F.US.YM",
    "MES": "F.US.MES", "MNQ": "F.US.MNQ", "MYM": "F.US.MYM",
    "GC": "F.US.GCE", "MGC": "F.US.MGC", "SI": "F.US.SIE", "SIL": "F.US.SIL",
}


def previous_quarter(contract_id: str) -> str:
    """CON.F.US.MES.U26 -> CON.F.US.MES.M26 (previous quarterly contract).
    Quarter codes: H=Mar, M=Jun, U=Sep, Z=Dec. Used to stitch enough daily-bar
    history for regime indicators during the ~6 weeks after each roll."""
    base, tail = contract_id.rsplit(".", 1)
    code, yy = tail[0], int(tail[1:])
    order = ["H", "M", "U", "Z"]
    i = order.index(code)
    if i == 0:
        return f"{base}.Z{yy - 1}"
    return f"{base}.{order[i - 1]}{yy}"


def resolve_contract(token: str, name: str) -> str:
    """Return the ACTIVE (front-month) contract id for a friendly name, asking the
    API. Falls back to the static CONTRACTS map if the lookup fails."""
    try:
        sym = SYMBOL_IDS.get(name)
        for c in get_contracts(token):
            if c.get("symbolId") == sym and c.get("activeContract"):
                return c["id"]
    except Exception:
        pass
    return CONTRACTS[name]


def get_contracts(token: str, live: bool = False) -> list[dict]:
    response = requests.post(
        CONTRACTS_URL,
        headers={
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"live": live},
    )
    response.raise_for_status()

    data = response.json()

    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"Contract search failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )

    return data["contracts"]


if __name__ == "__main__":
    token = get_session_token()
    contracts = get_contracts(token)

    if not contracts:
        print("No contracts found.")
    else:
        print(f"Found {len(contracts)} contracts:\n")
        for c in contracts:
            print(f"ID:   {c['id']}")
            print(f"Name: {c.get('name', 'N/A')}")
            print("-" * 30)
