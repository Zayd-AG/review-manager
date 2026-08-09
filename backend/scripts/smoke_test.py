"""Check that a running local Feedback Lens API can serve the demo routes.

Run after starting the Docker stack:
    python backend/scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_API_BASE_URL = "http://localhost:8000"
ROUTES = ("/health", "/ready", "/summary", "/dashboard?limit=1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    args = parser.parse_args()
    api_base_url = args.api_base_url.rstrip("/")

    for route in ROUTES:
        url = f"{api_base_url}{route}"
        try:
            with urlopen(url, timeout=10) as response:
                body = json.load(response)
        except HTTPError as error:
            raise SystemExit(f"FAIL {route}: HTTP {error.code}") from error
        except URLError as error:
            raise SystemExit(f"FAIL {route}: {error.reason}") from error
        print(f"PASS {route}: {response.status} ({type(body).__name__})")

    print("Local API smoke test passed.")


if __name__ == "__main__":
    main()
