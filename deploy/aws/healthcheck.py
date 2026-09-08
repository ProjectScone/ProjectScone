"""Container liveness only; does not contact a model or cloud backend."""

import json
import os
from urllib.request import urlopen


def main() -> None:
    port = int(os.environ.get("SCONE_PORT", "7437"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid healthcheck port")
    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as response:
        if response.status != 200 or json.loads(response.read(4096)) != {"ok": True}:
            raise RuntimeError("healthcheck failed")


if __name__ == "__main__":
    main()
