import requests
from auth import get_session_token

ACCOUNTS_URL = "https://api.topstepx.com/api/Account/search"


def get_accounts(token: str, only_active: bool = True) -> list[dict]:
    response = requests.post(
        ACCOUNTS_URL,
        headers={
            "accept": "text/plain",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={"onlyActiveAccounts": only_active},
    )
    response.raise_for_status()

    data = response.json()

    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"Account search failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )

    return data["accounts"]


if __name__ == "__main__":
    token = get_session_token()
    accounts = get_accounts(token)

    if not accounts:
        print("No accounts found.")
    else:
        for acc in accounts:
            print(f"ID:       {acc['id']}")
            print(f"Name:     {acc['name']}")
            print(f"Balance:  ${acc['balance']:,.2f}")
            print(f"Can Trade:{acc['canTrade']}")
            print(f"Visible:  {acc['isVisible']}")
            print("-" * 30)
