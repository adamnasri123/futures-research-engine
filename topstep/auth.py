import os
import requests
from dotenv import load_dotenv

load_dotenv()

AUTH_URL = "https://api.topstepx.com/api/Auth/loginKey"


def get_session_token() -> str:
    username = os.getenv("TOPSTEP_USERNAME")
    api_key = os.getenv("TOPSTEP_API_KEY")

    if not username or not api_key:
        raise ValueError("TOPSTEP_USERNAME and TOPSTEP_API_KEY must be set in .env")

    response = requests.post(
        AUTH_URL,
        headers={"accept": "text/plain", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
    )
    response.raise_for_status()

    data = response.json()

    if data.get("errorCode") != 0 or not data.get("success"):
        raise RuntimeError(
            f"Auth failed (errorCode={data.get('errorCode')}): {data.get('errorMessage') or 'no message returned'}"
        )

    return data["token"]


if __name__ == "__main__":
    token = get_session_token()
    print(f"Session token obtained successfully.")
    print(f"Token (first 40 chars): {token[:40]}...")
