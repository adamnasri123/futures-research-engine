import requests
from auth import get_session_token

CONTRACTS_URL = "https://api.topstepx.com/api/Contract/available"

# Friendly name -> contract ID. Update the IDs when contracts roll to a new month.
CONTRACTS = {
    # Mini stock futures
    "ES":  "CON.F.US.EP.M26",   # S&P 500 Mini
    "NQ":  "CON.F.US.ENQ.M26",  # Nasdaq Mini
    "YM":  "CON.F.US.YM.M26",   # Dow Mini
    # Micro stock futures
    "MES": "CON.F.US.MES.M26",  # Micro S&P 500
    "MNQ": "CON.F.US.MNQ.M26",  # Micro Nasdaq
    "MYM": "CON.F.US.MYM.M26",  # Micro Dow
    # Metals
    "GC":  "CON.F.US.GCE.Q26",  # Gold
    "MGC": "CON.F.US.MGC.Q26",  # Micro Gold
    "SI":  "CON.F.US.SIE.N26",  # Silver
    "SIL": "CON.F.US.SIL.N26",  # Micro Silver
}


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
